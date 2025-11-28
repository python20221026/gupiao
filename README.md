# 部署到 Streamlit Community Cloud（最快方式）

## 目录结构
- app.py（入口）
- indicators.py / strategies.py / backtest.py / broker.py
- requirements.txt
- runtime.txt（Python 版本）

## 步骤
1. 新建 GitHub 仓库（将本文件夹作为仓库根目录，确保 app.py、requirements.txt 在根目录）。
2. 推送代码到 GitHub。
3. 打开 Streamlit Community Cloud，选择 New app：
   - Repo：选择你的仓库
   - Branch：main（或你的分支）
   - Main file path：app.py
4. 部署。首次安装依赖需要几分钟。

## 注意
- 云端默认只支持“回测”和“实盘-模拟撮合”。Futu OpenAPI 与华泰 XTP 需要本地或有权限的服务器环境。
- akshare 访问第三方数据源，若网络限制导致拉取失败，可稍后重试或切换数据源方案。

