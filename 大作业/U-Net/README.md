# U-Net 医学图像分割项目

本项目是数字图像处理课程的大作业，任务是使用 U-Net 完成 ISBI 细胞边界分割实验，并在实验指导书代码基础上改进损失函数、输入分辨率和推理方式。

## 文件说明

- `code/01_reference_handout_code.md`：实验指导书对应的原始 U-Net 代码整理版。
- `code/02_best_pixel_weight_map_focal_tversky_overlap_tile.ipynb`：最终改进版代码，使用 `pixel-weight-map BCE + Focal Tversky`，并采用原分辨率 `patch + overlap-tile` 训练和推理。

## 最终方案

最终主实验使用：

- patch size：`256`
- center size / stride：`128`
- loss：`0.4 * pixel-weight-map BCE + 0.6 * Focal Tversky`
- optimizer：`SGD`
- batch size：`4`
- patches per image per epoch：`8`

## 运行说明

先参考 `01_reference_handout_code.md` 了解实验指导书的基础流程；复现实验最终结果时，运行 `02_best_pixel_weight_map_focal_tversky_overlap_tile.ipynb`。

notebook 默认在已解压 `Unet.zip` 的 ModelArts / Ascend 环境中运行。如果服务器尚未准备数据，需要先下载并解压实验数据包。
