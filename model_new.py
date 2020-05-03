import torch
import torch.nn as nn

import utils.utils as utils


class YOLODetection(nn.Module):
    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLODetection, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
                .permute(0, 1, 3, 4, 2).contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = utils.build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": utils.to_cpu(total_loss).item(),
                "x": utils.to_cpu(loss_x).item(),
                "y": utils.to_cpu(loss_y).item(),
                "w": utils.to_cpu(loss_w).item(),
                "h": utils.to_cpu(loss_h).item(),
                "conf": utils.to_cpu(loss_conf).item(),
                "cls": utils.to_cpu(loss_cls).item(),
                "cls_acc": utils.to_cpu(cls_acc).item(),
                "recall50": utils.to_cpu(recall50).item(),
                "recall75": utils.to_cpu(recall75).item(),
                "precision": utils.to_cpu(precision).item(),
                "conf_obj": utils.to_cpu(conf_obj).item(),
                "conf_noobj": utils.to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss


class YOLOv3(nn.Module):
    def __init__(self):
        super(YOLOv3, self).__init__()
        self.darknet53 = self.make_darknet53()
        self.conv_block1 = self.make_conv_block(1024, 1024)
        self.conv_final1 = self.make_conv(1024, 255, kernel_size=1, padding=0)

        self.upsample1 = self.make_upsample(1024, 256, scale_factor=2)
        self.conv_block2 = self.make_conv_block(768, 512)
        self.conv_final2 = self.make_conv(512, 255, kernel_size=1, padding=0)

        self.upsample2 = self.make_upsample(512, 128, scale_factor=2)
        self.conv_block3 = self.make_conv_block(384, 256)
        self.conv_final3 = self.make_conv(256, 255, kernel_size=1, padding=0)

    def forward(self, x):
        residual_output = {}

        # Darknet-53 forward
        for key, module in self.darknet53.items():
            module_type = key.split('_')[0]

            if module_type == 'conv':
                x = module(x)
            elif module_type == 'residual':
                block_iter = key.split('_')[-1]
                for i in range(int(block_iter[0])):
                    residual = x
                    out = module['block_{}'.format(i + 1)](x)
                    x = out + residual
                residual_output[key].append(x)

        # Yolov3 layer forward
        conv_b1 = self.conv_block1(residual_output['residual_5_4x'])
        scale1 = self.conv_final1(conv_b1)

        scale2 = self.upsample1(conv_b1)
        scale2 = torch.cat((scale2, residual_output['residual_4_8x']), dim=1)
        conv_b2 = self.conv_block2(scale2)
        scale2 = self.conv_final2(conv_b2)

        scale3 = self.upsample2(conv_b2)
        scale3 = torch.cat((scale3, residual_output['residual_3_8x']), dim=1)
        conv_b3 = self.conv_block3(scale3)
        scale3 = self.conv_final3(conv_b3)

        return scale1, scale2, scale3

    def make_darknet53(self):
        modules = nn.ModuleDict(
            {'conv_1': self.make_conv(3, 32, kernel_size=3),
             'conv_2': self.make_conv(32, 64, kernel_size=3, stride=2),
             'residual_1_1x': self.make_residual_block(in_channels=64, num_blocks=1),
             'conv_3': self.make_conv(64, 128, kernel_size=3, stride=2),
             'residual_2_2x': self.make_residual_block(in_channels=128, num_blocks=2),
             'conv_4': self.make_conv(128, 256, kernel_size=3, stride=2),
             'residual_3_8x': self.make_residual_block(in_channels=256, num_blocks=8),
             'conv_5': self.make_conv(256, 512, kernel_size=3, stride=2),
             'residual_4_8x': self.make_residual_block(in_channels=512, num_blocks=8),
             'conv_6': self.make_conv(512, 1024, kernel_size=3, stride=2),
             'residual_5_4x': self.make_residual_block(in_channels=1024, num_blocks=4)}
        )
        return modules

    def make_conv(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=1):
        modules = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5),
            nn.LeakyReLU(negative_slope=0.1)
        )
        return modules

    def make_conv_block(self, in_channels: int, out_channels: int):
        half_channels = out_channels // 2
        modules = nn.Sequential(
            self.make_conv(in_channels, half_channels, kernel_size=1, padding=0),
            self.make_conv(half_channels, out_channels, kernel_size=3),
            self.make_conv(out_channels, half_channels, kernel_size=1, padding=0),
            self.make_conv(half_channels, out_channels, kernel_size=3),
            self.make_conv(out_channels, half_channels, kernel_size=1, padding=0),
            self.make_conv(half_channels, out_channels, kernel_size=3)
        )
        return modules

    def make_residual_block(self, in_channels: int, num_blocks: int):
        half_channels = in_channels // 2
        block = nn.Sequential(
            self.make_conv(in_channels, half_channels, kernel_size=1, padding=0),
            self.make_conv(half_channels, in_channels, kernel_size=3)
        )

        modules = nn.ModuleDict()
        for i in range(num_blocks):
            modules['block_{}'.format(i + 1)] = block
        return modules

    def make_upsample(self, in_channels: int, out_channels: int, scale_factor: int):
        modules = nn.Sequential(
            self.make_conv(in_channels, out_channels, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=scale_factor, mode='nearest')
        )
        return modules


if __name__ == '__main__':
    model = YOLOv3()
    print(model)

    test = torch.rand([1, 3, 416, 416])
    y = model(test)
