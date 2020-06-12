import torch
import torch.nn as nn
import torch.utils.tensorboard
import numpy as np

import utils.utils


class YOLODetection(nn.Module):
    def __init__(self, anchors, image_size: int, num_classes: int):
        super(YOLODetection, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.image_size = image_size
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.ignore_thres = 0.5
        self.obj_scale = 1
        self.no_obj_scale = 100
        self.metrics = {}

    def forward(self, x, targets):
        device = torch.device('cuda' if x.is_cuda else 'cpu')

        num_batches = x.size(0)
        grid_size = x.size(2)

        # 출력값 형태 변환
        prediction = (
            x.view(num_batches, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
                .permute(0, 1, 3, 4, 2).contiguous()
        )

        # Get outputs
        cx = torch.sigmoid(prediction[..., 0])  # Center x
        cy = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Object confidence (objectness)
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Class prediction

        # Calculate offsets for each grid
        stride = self.image_size / grid_size
        grid_x = torch.arange(grid_size, dtype=torch.float, device=device).repeat(grid_size, 1).view(
            [1, 1, grid_size, grid_size])
        grid_y = torch.arange(grid_size, dtype=torch.float, device=device).repeat(grid_size, 1).t().view(
            [1, 1, grid_size, grid_size])
        scaled_anchors = torch.as_tensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors],
                                         dtype=torch.float, device=device)
        anchor_w = scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

        # Add offset and scale with anchors
        pred_boxes = torch.zeros_like(prediction[..., :4], device=device)
        pred_boxes[..., 0] = cx + grid_x
        pred_boxes[..., 1] = cy + grid_y
        pred_boxes[..., 2] = torch.exp(w) * anchor_w
        pred_boxes[..., 3] = torch.exp(h) * anchor_h

        pred = (pred_boxes.view(num_batches, -1, 4) * stride,
                pred_conf.view(num_batches, -1, 1),
                pred_cls.view(num_batches, -1, self.num_classes))
        output = torch.cat(pred, -1)

        if targets is None:
            return output, 0

        iou_scores, class_mask, obj_mask, no_obj_mask, tx, ty, tw, th, tcls, tconf = utils.utils.build_targets(
            pred_boxes=pred_boxes,
            pred_cls=pred_cls,
            target=targets,
            anchors=scaled_anchors,
            ignore_thres=self.ignore_thres,
            device=device
        )

        # Loss: Mask outputs to ignore non-existing objects (except with conf. loss)
        loss_x = self.mse_loss(cx[obj_mask], tx[obj_mask])
        loss_y = self.mse_loss(cy[obj_mask], ty[obj_mask])
        loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
        loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
        loss_bbox = loss_x + loss_y + loss_w + loss_h
        loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
        loss_conf_no_obj = self.bce_loss(pred_conf[no_obj_mask], tconf[no_obj_mask])
        loss_conf = self.obj_scale * loss_conf_obj + self.no_obj_scale * loss_conf_no_obj
        loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
        loss_layer = loss_bbox + loss_conf + loss_cls

        # Metrics
        conf50 = (pred_conf > 0.5).float()
        iou50 = (iou_scores > 0.5).float()
        iou75 = (iou_scores > 0.75).float()
        detected_mask = conf50 * class_mask * tconf
        cls_acc = 100 * class_mask[obj_mask].mean()
        conf_obj = pred_conf[obj_mask].mean()
        conf_no_obj = pred_conf[no_obj_mask].mean()
        precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
        recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
        recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

        # Write loss and metrics
        self.metrics = {
            "loss_x": loss_x.detach().cpu().item(),
            "loss_y": loss_y.detach().cpu().item(),
            "loss_w": loss_w.detach().cpu().item(),
            "loss_h": loss_h.detach().cpu().item(),
            "loss_bbox": loss_bbox.detach().cpu().item(),
            "loss_conf": loss_conf.detach().cpu().item(),
            "loss_cls": loss_cls.detach().cpu().item(),
            "loss_layer": loss_layer.detach().cpu().item(),
            "cls_acc": cls_acc.detach().cpu().item(),
            "conf_obj": conf_obj.detach().cpu().item(),
            "conf_no_obj": conf_no_obj.detach().cpu().item(),
            "precision": precision.detach().cpu().item(),
            "recall50": recall50.detach().cpu().item(),
            "recall75": recall75.detach().cpu().item()
        }

        return output, loss_layer


class ProposedYOLOv3(nn.Module):
    def __init__(self, image_size: int, num_classes: int):
        super(ProposedYOLOv3, self).__init__()
        anchors = {'scale1': [(350, 256), (262, 350), (378, 378)],
                   'scale2': [(209, 215), (326, 153), (178, 333)],
                   'scale3': [(182, 99), (131, 175), (120, 296)],
                   'scale4': [(51, 143), (92, 114), (74, 231)],
                   'scale5': [(26, 38), (31, 91), (69, 58)]}
        final_out_channel = 3 * (4 + 1 + num_classes)

        self.darknet53 = self.make_darknet53()

        self.scale2 = self.make_scale(512, 128, 256)
        self.scale3 = self.make_scale(1024, 256, 512)
        self.scale3_conv = self.make_conv(512, 256, kernel_size=1, stride=1, padding=0)
        self.scale4 = self.make_conv(1024, 256, kernel_size=1, stride=1, padding=0)
        self.scale5 = self.make_conv(512, 256, kernel_size=1, stride=1, padding=0)
        self.add_conv2 = self.make_conv(256, 256, kernel_size=3, stride=1, padding=1)
        self.add_conv3 = self.make_conv(256, 256, kernel_size=3, stride=1, padding=1)
        self.add_conv4 = self.make_conv(256, 256, kernel_size=3, stride=1, padding=1)
        self.add_b_conv2 = self.make_conv(256, 256, kernel_size=3, stride=1, padding=1)
        self.add_b_conv3 = self.make_conv(256, 256, kernel_size=3, stride=1, padding=1)
        self.add_b_conv4 = self.make_conv(256, 256, kernel_size=3, stride=1, padding=1)
        self.add_b_conv5 = self.make_conv(256, 256, kernel_size=3, stride=1, padding=1)
        self.add_b_conv6 = self.make_conv(256, 256, kernel_size=3, stride=1, padding=1)
        self.upsample2 = nn.Upsample(size=7, mode='nearest')
        self.upsample3 = nn.Upsample(size=13, mode='nearest')
        self.upsample4 = nn.Upsample(size=26, mode='nearest')
        self.upsample5 = nn.Upsample(size=52, mode='nearest')
        self.downsample2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.downsample3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.downsample4 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.downsample5 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.conv_final2 = nn.Conv2d(256, final_out_channel, kernel_size=1, stride=1, padding=0)
        self.conv_final3 = nn.Conv2d(256, final_out_channel, kernel_size=1, stride=1, padding=0)
        self.conv_final4 = nn.Conv2d(256, final_out_channel, kernel_size=1, stride=1, padding=0)
        self.conv_final5 = nn.Conv2d(256, final_out_channel, kernel_size=1, stride=1, padding=0)
        self.conv_final6 = nn.Conv2d(256, final_out_channel, kernel_size=1, stride=1, padding=0)

        self.yolo_layer2 = YOLODetection(anchors['scale1'], image_size, num_classes)
        self.yolo_layer3 = YOLODetection(anchors['scale2'], image_size, num_classes)
        self.yolo_layer4 = YOLODetection(anchors['scale3'], image_size, num_classes)
        self.yolo_layer5 = YOLODetection(anchors['scale4'], image_size, num_classes)
        self.yolo_layer6 = YOLODetection(anchors['scale5'], image_size, num_classes)

        self.yolo_layers = [self.yolo_layer2, self.yolo_layer3,
                            self.yolo_layer4, self.yolo_layer5, self.yolo_layer6]

    def forward(self, x, targets=None):
        loss = 0
        residual_output = {}

        # Darknet-53 forward
        with torch.no_grad():
            for key, module in self.darknet53.items():
                module_type = key.split('_')[0]

                if module_type == 'conv':
                    x = module(x)
                elif module_type == 'residual':
                    out = module(x)
                    x += out
                    if key == 'residual_3_8' or key == 'residual_4_8' or key == 'residual_5_4':
                        residual_output[key] = x

        # Yolov3 layer forward
        scale6 = residual_output['residual_3_8']
        scale5 = self.scale5(residual_output['residual_4_8'])
        scale4 = self.scale4(residual_output['residual_5_4'])
        scale3_temp = self.scale3(residual_output['residual_5_4'])
        scale3 = self.scale3_conv(scale3_temp)
        scale2 = self.scale2(scale3_temp)

        add2 = self.add_conv2(self.upsample2(scale2) + scale3)
        add3 = self.add_conv3(self.upsample3(add2) + scale4)
        add4 = self.add_conv4(self.upsample4(add3) + scale5)

        add_b6 = self.add_b_conv6(self.upsample5(add4) + scale6)
        add_b5 = self.add_b_conv5(self.downsample5(add_b6) + add4)
        add_b4 = self.add_b_conv4(self.downsample4(add_b5) + add3)
        add_b3 = self.add_b_conv3(self.downsample3(add_b4) + add2)
        add_b2 = self.add_b_conv2(self.downsample2(add_b3) + scale2)

        final2 = self.conv_final2(add_b2)
        final3 = self.conv_final3(add_b3)
        final4 = self.conv_final4(add_b4)
        final5 = self.conv_final5(add_b5)
        final6 = self.conv_final6(add_b6)

        yolo_output2, layer_loss = self.yolo_layer2(final2, targets)
        loss += layer_loss
        yolo_output3, layer_loss = self.yolo_layer3(final3, targets)
        loss += layer_loss
        yolo_output4, layer_loss = self.yolo_layer4(final4, targets)
        loss += layer_loss
        yolo_output5, layer_loss = self.yolo_layer5(final5, targets)
        loss += layer_loss
        yolo_output6, layer_loss = self.yolo_layer6(final6, targets)
        loss += layer_loss

        yolo_outputs = [yolo_output2, yolo_output3,
                        yolo_output4, yolo_output5, yolo_output6]
        yolo_outputs = torch.cat(yolo_outputs, 1).detach().cpu()
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def make_darknet53(self):
        modules = nn.ModuleDict()

        modules['conv_1'] = self.make_conv(3, 32, kernel_size=3, requires_grad=False)
        modules['conv_2'] = self.make_conv(32, 64, kernel_size=3, stride=2, requires_grad=False)
        modules['residual_1_1'] = self.make_residual_block(in_channels=64)
        modules['conv_3'] = self.make_conv(64, 128, kernel_size=3, stride=2, requires_grad=False)
        modules['residual_2_1'] = self.make_residual_block(in_channels=128)
        modules['residual_2_2'] = self.make_residual_block(in_channels=128)
        modules['conv_4'] = self.make_conv(128, 256, kernel_size=3, stride=2, requires_grad=False)
        modules['residual_3_1'] = self.make_residual_block(in_channels=256)
        modules['residual_3_2'] = self.make_residual_block(in_channels=256)
        modules['residual_3_3'] = self.make_residual_block(in_channels=256)
        modules['residual_3_4'] = self.make_residual_block(in_channels=256)
        modules['residual_3_5'] = self.make_residual_block(in_channels=256)
        modules['residual_3_6'] = self.make_residual_block(in_channels=256)
        modules['residual_3_7'] = self.make_residual_block(in_channels=256)
        modules['residual_3_8'] = self.make_residual_block(in_channels=256)
        modules['conv_5'] = self.make_conv(256, 512, kernel_size=3, stride=2, requires_grad=False)
        modules['residual_4_1'] = self.make_residual_block(in_channels=512)
        modules['residual_4_2'] = self.make_residual_block(in_channels=512)
        modules['residual_4_3'] = self.make_residual_block(in_channels=512)
        modules['residual_4_4'] = self.make_residual_block(in_channels=512)
        modules['residual_4_5'] = self.make_residual_block(in_channels=512)
        modules['residual_4_6'] = self.make_residual_block(in_channels=512)
        modules['residual_4_7'] = self.make_residual_block(in_channels=512)
        modules['residual_4_8'] = self.make_residual_block(in_channels=512)
        modules['conv_6'] = self.make_conv(512, 1024, kernel_size=3, stride=2, requires_grad=False)
        modules['residual_5_1'] = self.make_residual_block(in_channels=1024)
        modules['residual_5_2'] = self.make_residual_block(in_channels=1024)
        modules['residual_5_3'] = self.make_residual_block(in_channels=1024)
        modules['residual_5_4'] = self.make_residual_block(in_channels=1024)
        return modules

    def make_scale(self, in_channels: int, reduce_channels: int, out_channels: int, last=False):
        module1 = self.make_conv(in_channels, reduce_channels, kernel_size=1, stride=1, padding=0)
        if last:
            module2 = self.make_conv(reduce_channels, out_channels, kernel_size=3, stride=1, padding=0)
        else:
            module2 = self.make_conv(reduce_channels, out_channels, kernel_size=3, stride=2, padding=1)
        modules = nn.Sequential(module1, module2)
        return modules

    def make_conv(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=1, requires_grad=True):
        module1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        module2 = nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5)
        if not requires_grad:
            for param in module1.parameters():
                param.requires_grad_(False)
            for param in module2.parameters():
                param.requires_grad_(False)

        modules = nn.Sequential(module1, module2, nn.LeakyReLU(negative_slope=0.1))
        return modules

    def make_residual_block(self, in_channels: int):
        half_channels = in_channels // 2
        block = nn.Sequential(
            self.make_conv(in_channels, half_channels, kernel_size=1, padding=0, requires_grad=False),
            self.make_conv(half_channels, in_channels, kernel_size=3, requires_grad=False)
        )
        return block

    # Load original weights file
    def load_darknet_weights(self, weights_path: str):
        # Open the weights file
        with open(weights_path, "rb") as f:
            _ = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values (0~2: version, 3~4: seen)
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        ptr = 0
        # Load Darknet-53 weights
        for key, module in self.darknet53.items():
            module_type = key.split('_')[0]

            if module_type == 'conv':
                ptr = self.load_bn_weights(module[1], weights, ptr)
                ptr = self.load_conv_weights(module[0], weights, ptr)

            elif module_type == 'residual':
                for i in range(2):
                    ptr = self.load_bn_weights(module[i][1], weights, ptr)
                    ptr = self.load_conv_weights(module[i][0], weights, ptr)

    # Load BN bias, weights, running mean and running variance
    def load_bn_weights(self, bn_layer, weights, ptr: int):
        num_bn_biases = bn_layer.bias.numel()

        # Bias
        bn_biases = torch.from_numpy(weights[ptr: ptr + num_bn_biases]).view_as(bn_layer.bias)
        bn_layer.bias.data.copy_(bn_biases)
        ptr += num_bn_biases
        # Weight
        bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases]).view_as(bn_layer.weight)
        bn_layer.weight.data.copy_(bn_weights)
        ptr += num_bn_biases
        # Running Mean
        bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases]).view_as(bn_layer.running_mean)
        bn_layer.running_mean.data.copy_(bn_running_mean)
        ptr += num_bn_biases
        # Running Var
        bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases]).view_as(bn_layer.running_var)
        bn_layer.running_var.data.copy_(bn_running_var)
        ptr += num_bn_biases

        return ptr

    # Load convolution weights
    def load_conv_weights(self, conv_layer, weights, ptr: int):
        num_weights = conv_layer.weight.numel()

        conv_weights = torch.from_numpy(weights[ptr: ptr + num_weights])
        conv_weights = conv_weights.view_as(conv_layer.weight)
        conv_layer.weight.data.copy_(conv_weights)
        ptr += num_weights

        return ptr


if __name__ == '__main__':
    model = ProposedYOLOv3(image_size=416, num_classes=80)
    model.load_darknet_weights('../weights/darknet53.conv.74')
    print(model)

    test = torch.rand([2, 3, 416, 416])
    y = model(test)

    writer = torch.utils.tensorboard.SummaryWriter('../logs')
    writer.add_graph(model, test)
    writer.close()
