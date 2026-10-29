# Contributing / 参与开发

Use Apple Silicon macOS for full numerical tests. The web frontend is plain HTML/CSS/JavaScript; no frontend build step is needed.

```bash
bash scripts/setup.sh
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
node tests/test_i18n.cjs
```

No model weights are required by these tests. `tests/web_smoke.py` is separate: it needs a running, idle Web service and the default model, and performs real inference. Do not run it against an active user session.

Keep UI translations in `web/static/i18n-catalog.js`; localize UI labels without changing user prompts, model output, or third-party recipe prose. Update English and Chinese README sections when behavior changes. New model support should document chat, steering, LoRA, and recipe compatibility separately.

Please include reproduction steps, model repository/revision/format, macOS and hardware details, and relevant error text in bug reports. Remove private prompts, local paths and tokens before sharing logs. Do not commit models, adapter weights, binaries, caches, experiment exports, or private datasets.

完整数值测试需要 Apple Silicon；日常单元测试不下载模型。修改功能时同步更新 README 中英文部分及相关文档。新模型应分别验证聊天、干预、LoRA 和配方兼容性；下载成功不等于已支持全部能力。提交问题时请附复现步骤与版本信息，并去除个人内容。权重及本地实验产物不得提交。
