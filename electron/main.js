'use strict';

const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn, execSync } = require('child_process');
const http = require('http');

// ─── Path Resolution ────────────────────────────────────────────────────────
const APP_ROOT = app.isPackaged
  ? path.join(process.resourcesPath, 'app')
  : path.join(__dirname, '..');

const CONFIG_PATH = path.join(APP_ROOT, 'configs', 'runtime', 'app_config.json');
const REQUIREMENTS_PATH = path.join(APP_ROOT, 'requirements.txt');
const FRONTEND_INDEX = path.join(APP_ROOT, 'visualization', 'frontend', 'dist', 'index.html');
const SETUP_HTML = path.join(__dirname, 'setup.html');

// venv paths — placed next to the app folder, not inside it,
// to avoid path issues on Windows (app/.venv vs app.venv confusion)
const VENV_BASE = app.isPackaged ? process.resourcesPath : path.join(__dirname, '..');
const VENV_DIR = path.join(VENV_BASE, 'venv');
const VENV_PYTHON = path.join(VENV_DIR, 'Scripts', 'python.exe');
const VENV_PIP = path.join(VENV_DIR, 'Scripts', 'pip.exe');

// Backend port
const BACKEND_PORT = 8000;
const HEALTH_URL = `http://127.0.0.1:${BACKEND_PORT}/health`;

// ─── State ───────────────────────────────────────────────────────────────────
let mainWindow = null;
let setupWindow = null;
let backendProcess = null;
let splashWindow = null;

// ─── Config Helpers ──────────────────────────────────────────────────────────
function readConfig() {
  try {
    if (fs.existsSync(CONFIG_PATH)) {
      return JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf-8'));
    }
  } catch (e) {
    console.error('Failed to read config:', e);
  }
  return null;
}

function writeConfig(cfg) {
  const dir = path.dirname(CONFIG_PATH);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(CONFIG_PATH, JSON.stringify(cfg, null, 2), 'utf-8');
}

// ─── Splash Window ───────────────────────────────────────────────────────────
function createSplashWindow(message) {
  splashWindow = new BrowserWindow({
    width: 480,
    height: 280,
    frame: false,
    resizable: false,
    center: true,
    alwaysOnTop: true,
    webPreferences: { nodeIntegration: true, contextIsolation: false },
  });

  const html = `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #0f1117;
    color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    height: 100vh; user-select: none;
  }
  h1 { font-size: 28px; font-weight: 700; color: #60a5fa; margin-bottom: 8px; }
  .subtitle { font-size: 13px; color: #94a3b8; margin-bottom: 32px; }
  #msg { font-size: 14px; color: #cbd5e1; }
  .spinner {
    width: 36px; height: 36px;
    border: 3px solid #1e293b;
    border-top-color: #60a5fa;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-bottom: 16px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
  <h1>NeuroLens</h1>
  <div class="subtitle">LLM Safety Neuron Analysis</div>
  <div class="spinner"></div>
  <div id="msg">${message}</div>
  <script>
    const { ipcRenderer } = require('electron');
    ipcRenderer.on('splash-message', (_, text) => {
      document.getElementById('msg').textContent = text;
    });
  </script>
</body>
</html>`;

  splashWindow.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(html)}`);
}

function updateSplash(message) {
  if (splashWindow && !splashWindow.isDestroyed()) {
    splashWindow.webContents.send('splash-message', message);
  }
}

function closeSplash() {
  if (splashWindow && !splashWindow.isDestroyed()) {
    splashWindow.close();
    splashWindow = null;
  }
}

// ─── Setup Window ────────────────────────────────────────────────────────────
function createSetupWindow() {
  setupWindow = new BrowserWindow({
    width: 600,
    height: 560,
    resizable: false,
    center: true,
    title: 'NeuroLens — 初始配置',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  setupWindow.loadFile(SETUP_HTML);
  setupWindow.on('closed', () => { setupWindow = null; });
}

// ─── Main Window ─────────────────────────────────────────────────────────────
function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1024,
    minHeight: 600,
    title: 'NeuroLens',
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.loadFile(FRONTEND_INDEX);

  mainWindow.once('ready-to-show', () => {
    closeSplash();
    mainWindow.show();
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
    stopBackend();
  });

  // Open external links in browser
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });
}

// ─── venv & Dependencies ─────────────────────────────────────────────────────
function getPython() {
  // Prefer system python3/python
  for (const cmd of ['python', 'python3']) {
    try {
      const version = execSync(`${cmd} --version`, { encoding: 'utf-8' });
      if (version.includes('Python 3')) return cmd;
    } catch (_) {}
  }
  return null;
}

function ensureVenv(pythonCmd) {
  if (!fs.existsSync(VENV_PYTHON)) {
    updateSplash('正在创建 Python 虚拟环境...');
    execSync(`${pythonCmd} -m venv "${VENV_DIR}"`, { stdio: 'inherit' });
  }
}

function installDependencies() {
  updateSplash('正在安装 Python 依赖（首次安装需要几分钟）...');
  // Install visualization backend deps first (lightweight, fast)
  const vizReqs = path.join(APP_ROOT, 'visualization', 'backend', 'requirements.txt');
  execSync(`"${VENV_PIP}" install -r "${vizReqs}" --quiet`, { stdio: 'inherit' });
  // Install full requirements (torch etc.)
  execSync(`"${VENV_PIP}" install -r "${REQUIREMENTS_PATH}" --quiet`, { stdio: 'inherit' });
  // Mark installation complete
  fs.writeFileSync(path.join(VENV_DIR, '.installed'), new Date().toISOString());
}

function isDepsInstalled() {
  return fs.existsSync(path.join(VENV_DIR, '.installed'));
}

// ─── Backend Process ─────────────────────────────────────────────────────────
function startBackend(cfg) {
  const env = {
    ...process.env,
    PYTHONPATH: APP_ROOT,
    NEUROLENS_OUTPUTS_DIR: cfg.outputs_dir || '',
    LLM_LOCAL_PATH: cfg.llm_path || '',
    GUARD_LOCAL_PATH: cfg.guard_path || '',
    CUDA_VISIBLE_DEVICES: cfg.cuda_device || '0',
  };

  const backendScript = path.join(APP_ROOT, 'visualization', 'backend', 'main.py');
  backendProcess = spawn(VENV_PYTHON, ['-m', 'uvicorn', 'main:app',
    '--host', '127.0.0.1',
    '--port', String(BACKEND_PORT),
    '--app-dir', path.join(APP_ROOT, 'visualization', 'backend'),
  ], { env, cwd: APP_ROOT });

  backendProcess.stdout.on('data', (d) => console.log('[backend]', d.toString().trim()));
  backendProcess.stderr.on('data', (d) => console.error('[backend]', d.toString().trim()));
  backendProcess.on('exit', (code) => {
    console.log(`[backend] exited with code ${code}`);
    backendProcess = null;
  });
}

function stopBackend() {
  if (backendProcess) {
    backendProcess.kill();
    backendProcess = null;
  }
}

// ─── Health Check ─────────────────────────────────────────────────────────────
function waitForBackend(maxSeconds = 60) {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    let attempt = 0;
    const check = () => {
      attempt++;
      updateSplash(`正在等待后端服务启动... (${attempt})`);
      http.get(HEALTH_URL, (res) => {
        if (res.statusCode === 200) return resolve();
        schedule();
      }).on('error', () => {
        if (Date.now() - start > maxSeconds * 1000) {
          return reject(new Error('Backend failed to start within timeout'));
        }
        schedule();
      });
    };
    const schedule = () => setTimeout(check, 1000);
    check();
  });
}

// ─── IPC Handlers ────────────────────────────────────────────────────────────
ipcMain.handle('get-config', () => readConfig());

ipcMain.handle('save-config', (_, cfg) => {
  writeConfig(cfg);
  return true;
});

ipcMain.handle('select-folder', async () => {
  const result = await dialog.showOpenDialog(setupWindow || mainWindow, {
    properties: ['openDirectory'],
  });
  return result.canceled ? null : result.filePaths[0];
});

ipcMain.handle('setup-complete', async (_, cfg) => {
  writeConfig(cfg);
  if (setupWindow && !setupWindow.isDestroyed()) setupWindow.close();
  await launchApp(cfg);
});

// ─── App Launch Sequence ─────────────────────────────────────────────────────
async function launchApp(cfg) {
  createSplashWindow('正在初始化...');

  try {
    const python = getPython();
    if (!python) {
      dialog.showErrorBox('未找到 Python',
        '请先安装 Python 3.9 或更高版本，并将其加入系统 PATH。\n\n下载地址：https://python.org');
      app.quit();
      return;
    }

    ensureVenv(python);

    if (!isDepsInstalled()) {
      installDependencies();
    }

    updateSplash('正在启动后端服务...');
    startBackend(cfg);
    await waitForBackend(60);

    createMainWindow();
  } catch (err) {
    closeSplash();
    dialog.showErrorBox('启动失败', err.message);
    app.quit();
  }
}

// ─── App Lifecycle ────────────────────────────────────────────────────────────
app.whenReady().then(() => {
  const cfg = readConfig();

  if (!cfg || !cfg.setup_completed) {
    createSetupWindow();
  } else {
    launchApp(cfg);
  }
});

app.on('window-all-closed', () => {
  stopBackend();
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (mainWindow === null && !setupWindow) {
    const cfg = readConfig();
    if (cfg && cfg.setup_completed) launchApp(cfg);
  }
});

app.on('before-quit', () => stopBackend());
