# 首次安装、运行与停止

本项目运行目标为 Apple Silicon macOS。仓库不包含模型、运行时二进制、适配器和实验历史。

## 安装

先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，再运行：

```bash
git clone https://github.com/IceThunder/qwen-lab.git
cd qwen-lab
bash scripts/setup.sh
bash scripts/download.sh
bash scripts/web.sh
```

下载默认使用 [Qwen3.5-4B MLX 4-bit](https://huggingface.co/mlx-community/Qwen3.5-4B-4bit)，固定本次解析出的提交，并保存到 `models/qwen3.5-4b/`。下载中断后重新运行 `download.sh` 续传；已安装目录不会覆盖。官方原始模型见 [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)。

也可跳过下载命令，直接启动 Web，通过“模型管理”添加模型。默认 MLX 仓库首次安装使用同一目录标识。GGUF 需先安装 llama.cpp，例如 `brew install llama.cpp`，仅支持聊天。

## 启动与停止

打开 http://127.0.0.1:7860/ 。终端保持运行，按 Ctrl+C 停止 Web 和模型子进程；关闭网页不会停止服务。首次发送才加载模型，下载模型不会自动加载。

端口占用时使用 `bash scripts/web.sh --port 7861`。浏览器会话按地址/端口分开保存。日常启动只需在项目根目录运行 `bash scripts/web.sh`，无需重复安装或下载。

## 使用

1. 选择模型和提示词模板，按需编辑系统提示词、前缀与后缀。
2. 输入问题，点击发送，或按 Cmd/Ctrl+Enter；完成后继续追问。
3. 生成上限为 1–4096 token，上下文预算为 2048/4096/8192；输入加输出预留不能超出预算。思考内容也占生成预算。
4. 如需神经元实验，展开默认隐藏的控制台，添加干预组并开启无干预对照。收起控制台仍会保留已启用的干预。
5. 顶部 English / 中文按钮即时切换界面，刷新后记住语言。已有输入、参数与回答不会被翻译或清空。
6. 实验会自动保存在浏览器；重要内容请导出 JSON。切换模型、新建会话或导入配方前按界面提示保存。

停止生成只取消当前任务。GGUF 无法使用当前 MLX 干预和 LoRA。未完成、错误或取消的回答不进入后续上下文。

详见 [Web 使用说明](WEB_UI.md)、[模型管理](MODEL_MANAGER_ZH.md)、[Neuronary 导入](NEURONARY_ZH.md) 和 [运行、API、训练](USAGE_ZH.md)。
