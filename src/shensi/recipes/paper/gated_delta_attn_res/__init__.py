"""GDAR = Gated Delta Attention Residuals 的训练配方（论文复现件）。

在 PreNorm transformer 的**深度轴**（残差流）上做「带门控的 delta 规则」：
把深度记忆当作可编辑的状态（decay / erase / write 三门 + 闭式解更新 + 白化读）。
出处与完整实验报告见 gdar_package（README_PACKAGE.md / code/train/RECIPE.md）；
本配方把其中的模型实现、训练链路与消融矩阵落进 shensi 仓库的配方约定里。
"""
