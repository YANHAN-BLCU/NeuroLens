; NeuroLens Windows 安装包 — 需先安装 Inno Setup 6：https://jrsoftware.org/isinfo.php
; 构建前请先执行 PyInstaller（NeuroLens.spec 将产物输出到仓库根目录）
; 版本宏来自 _AppVersion.iss（由 build_installer.ps1 根据仓库根 VERSION 生成）

#include "_AppVersion.iss"

#define MyAppName "NeuroLens"
#define MyAppPublisher "NeuroLens"
#define MyAppExeName "NeuroLens.exe"
#define DistDir "..\visualization\backend\dist\NeuroLens"
#define ProjectDir ".."

[Setup]
AppId={{B4E8C1A2-9F3D-4E7B-8C6D-1A2B3C4D5E6F}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName=C:\Program Files\{#MyAppName}
DisableDirPage=no
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=Output
OutputBaseFilename=NeuroLens_Setup_{#MyAppVersion}
SetupIconFile={#ProjectDir}\pic\installer.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
VersionInfoVersion={#MyVersionInfo}
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription={#MyAppName} 安装程序
VersionInfoProductName={#MyAppName}
VersionInfoProductVersion={#MyAppVersion}
LicenseFile=Disclaimer.txt
; 使用下方「任务」控制开始菜单/桌面快捷方式，不再单独显示「程序组」页
DisableProgramGroupPage=yes

[Languages]
Name: "chs"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
Name: "enu"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "startmenu"; Description: "添加至开始菜单"; GroupDescription: "快捷方式:"; Flags: unchecked
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "快捷方式:"; Flags: checkedonce
Name: "launch"; Description: "安装完成后运行 NeuroLens"; GroupDescription: "安装后:"; Flags: checkedonce


[Files]
; 主程序与 _internal（不排除 models 目录）
Source: "{#ProjectDir}\*"; DestDir: "{app}"; \
    Excludes: "\outputs\*,\data\*,\dev_env\*,\vis_backup\*,\build\*,installer\Output\*,visualization\backend\dist\*,visualization\backend\build\*,unins000.*,*.pyc,__pycache__\*,.git\*"; \
    Flags: ignoreversion recursesubdirs createallsubdirs

; 模型 1：仓库根目录 models\（大语言模型等，完整目录，不做排除）
Source: "{#ProjectDir}\models\*"; \
    DestDir: "{app}\models"; \
    Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist

; 模型 2：脱敏 NER 模型 chinese-ner-per-addr-rbt3（完整目录，不做排除）
Source: "{#ProjectDir}\_internal\Desensization-dashboard\models\chinese-ner-per-addr-rbt3\*"; \
    DestDir: "{app}\_internal\Desensization-dashboard\models\chinese-ner-per-addr-rbt3"; \
    Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: startmenu
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "启动 {#MyAppName}"; Flags: nowait postinstall skipifsilent; Tasks: launch
