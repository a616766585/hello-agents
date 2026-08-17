# Hello Agents

用于实现和比较不同智能体范式的 Python 示例项目。

## 目录

```text
common/          跨智能体复用的组件（目前为 LLM 客户端）
ReAct/           ReAct 智能体、工具与运行入口
PlanAndSolve/    Plan-and-Solve 智能体预留目录
tests/           离线单元测试
```

## 配置

将 `.env.example` 复制为 `.env`，并填写 LLM 与 SerpApi 的配置。`LLM_BASE_URL` 应使用服务商控制台提供的当前 OpenAI 兼容 Base URL，通常以 `/v1` 结尾，不要包含 `/chat/completions`。

## 运行 ReAct 示例

```bash
python -m ReAct.Main
```

## 测试

```bash
python -m unittest discover -s tests -v
```
