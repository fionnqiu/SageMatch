# SageMatch（知弈）

岗位定制化模拟面试训练系统。项目名仅作代号，界面与文案一律使用「模拟面试」。

第一期：像素还原会话舱 + 模拟面试（入口 / 进行中 / 复盘），后端跑通「岗位提交 → 出题 → 文字面试 → 复盘」。管理端先做可进的壳。语音 / RAG 评测后置。

## 本地启动

需要：Python 3.11、Node 24、本机 PostgreSQL（库名 `sagematch`）。复制 `.env.example` 为 `.env` 并填密码与 LLM 密钥。

```powershell
# 后端
cd backend
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m uvicorn app.main:app --reload --port 8000

# 前端（另开一个终端）
cd frontend
npm install
npm run dev
```

- 用户端：http://localhost:5173
- 管理端：http://localhost:5173/admin
- API 文档：http://127.0.0.1:8000/docs
