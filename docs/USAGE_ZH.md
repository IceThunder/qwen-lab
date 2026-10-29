# 命令行、API、干预与 LoRA

以下命令均在 `qwen-lab` 根目录执行，先完成 `bash scripts/setup.sh` 和 `bash scripts/download.sh`。命令行与训练默认使用 `models/qwen3.5-4b/weights`；模型管理中的其他模型不会自动获得训练配置。

## 单轮运行与干预

```bash
bash scripts/run.sh --prompt "请解释激活值是什么。" --max-tokens 128
bash scripts/run.sh --prompt "解释梯度下降。" --thinking --max-tokens 512
bash scripts/intervene.sh --layer 16 --neurons 0,1,2,3 --scales 0,1,2
```

CLI 默认贪心生成、关闭思考；输入限制 2048 token，`run.sh` 输出限制 1–1024。Web 提供不同的预算范围。干预发生在 MLP 中间激活进入 down_proj 前；CLI 演示乘法，Web 还支持残差归一化加法。报告写入 `outputs/qwen3.5-4b/`，不会改写权重。

## 独立 OpenAI 兼容 API

```bash
bash scripts/serve.sh
```

服务监听 http://127.0.0.1:8080/v1 。另开终端进入项目根目录：

```bash
.venv/bin/python scripts/call_api.py --prompt "你好，请介绍自己。" --max-tokens 128
```

Web 不依赖这个 API 服务。内存有限时不要同时运行多个模型服务；在启动终端按 Ctrl+C 停止。

## LoRA 微调

`configs/qwen3.5-4b/lora.yaml` 默认只训练最后 2 层的 MLP LoRA，rank=4、batch=1、序列长度 256，并启用梯度检查点。示例数据 `data/example/{train,valid,test}.jsonl` 是合成的小样本，用于验证流程，不代表实用微调数据。

先停止 Web/API，使用独立的适配器目录做短训练：

```bash
bash scripts/train.sh --iters 2 --val-batches 1 --steps-per-eval 2 \
  --save-every 2 --adapter-path adapters/qwen3.5-4b/my-experiment
bash scripts/evaluate.sh --adapter-path adapters/qwen3.5-4b/my-experiment
bash scripts/run.sh --adapter adapters/qwen3.5-4b/my-experiment \
  --prompt "什么是激活值？" --max-tokens 128
```

正式训练应准备自己的数据、修改 YAML 和训练迭代数，并用独立测试集评估。默认 `evaluate.sh` 不加载训练配置，避免误启动训练。不要将不同基础模型的适配器混用。

重启 Web 后可在 LoRA 下拉列表中选择 `adapters/<模型ID>/<实验名>/` 中的适配器，需要同时存在 `adapters.safetensors` 和 `adapter_config.json`。神经元干预是临时修改激活，LoRA 才会产生新的训练参数文件。

## 本地产物

- `models/`：权重、来源与校验记录；全部不提交 Git。
- `adapters/`：训练结果；`outputs/`：干预报告；`logs/`：运行日志。
- `work/`：未完成下载与临时数据，续传完成前不要清理。
- `.venv/`、`.runtime/`、`.cache/`：本地环境、解释器与缓存。

自有训练数据放在 `data/` 下的其他目录（Git 默认排除），勿覆盖仓库演示数据后误提交隐私内容。
