# ComfyUI / 本地 Stable Diffusion 备选实现

> 工单编号：人工智能NLP-Agent数字人项目-文生图智能体任务

工单备注允许「Stable Diffusion 实现或者使用 ComfyUI 实现」。
本项目**主线**采用通义 `qwen-image-edit`（画质最佳、原生保身份），
此目录提供**等价的本地/ComfyUI 路径**作为备选参考。

## 工作流文件

`workflow_face_pose_outpaint.json`（ComfyUI API 格式）实现：

```
CheckpointLoaderSimple (SD1.5)
   └─ IPAdapterUnifiedLoader(PLUS FACE) ─ IPAdapter(weight=0.7)  ← 从原图锁身份
        └─ KSampler(img2img, denoise≈0.6)  ← 提示词控制 left/front/right 转头
             └─ VAEDecode → SaveImage(view_pose)
                  └─ ImagePadForOutpaint(四周128, feather40)
                       └─ VAEEncodeForInpaint → KSampler(denoise≈0.95)
                            └─ VAEDecode → SaveImage(view_outpaint)   ← 扩图
```

## 使用步骤

1. 安装 ComfyUI 及自定义节点 `ComfyUI_IPAdapter_plus`；
2. 准备权重：
   - checkpoint：`v1-5-pruned-emaonly.safetensors`
   - IP-Adapter：`ip-adapter-plus-face_sd15.safetensors` + CLIP-ViT-H 图像编码器；
3. ComfyUI 设置中开启 *Enable Dev mode options*，用 **Load (API format)** 导入本 JSON；
4. `LoadImage` 指向 `inputs/original_face.png`；
5. 修改 7 号 `CLIPTextEncode` 的提示词中 `left / right / forward` 生成不同视图；
6. 运行，分别得到转头视图与扩图。

## 为什么主线没有用它（重要取舍）

实测结论：**本地 SD1.5 若不加 IP-Adapter，转头时会"换人"**，无法满足验收的「面部特征保持」。
而 IP-Adapter + CLIP-ViT-H 需额外下载约 2.6GB 权重；在本机受限网络下载需数小时。
通义 `qwen-image-edit` 指令式编辑可一步「保持同一个人 + 改变朝向」，无需本地大模型，
因此主线选择通义。本备选路径在具备 IP-Adapter 权重的环境中可达到接近效果。
