'use strict';

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  getConfig: () => ipcRenderer.invoke('get-config'),
  saveConfig: (cfg) => ipcRenderer.invoke('save-config', cfg),
  selectFolder: () => ipcRenderer.invoke('select-folder'),
  setupComplete: (cfg) => ipcRenderer.invoke('setup-complete', cfg),
  onBackendReady: (cb) => ipcRenderer.on('backend-ready', (_, ...args) => cb(...args)),
  onSplashMessage: (cb) => ipcRenderer.on('splash-message', (_, msg) => cb(msg)),
});
