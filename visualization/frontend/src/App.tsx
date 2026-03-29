// NeuroLens Visualization - Main App Component

import React from 'react';
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  CircularProgress,
  Alert,
  IconButton,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Timeline as TimelineIcon,
  Layers as LayersIcon,
  Hub as HubIcon,
  ListAlt as ListAltIcon,
  ShowChart as ShowChartIcon,
  GridView as GridViewIcon,
  Menu as MenuIcon,
} from '@mui/icons-material';

import { useStore } from './store';
import { ControlPanel } from './components/ControlPanel';
import { MetricView } from './components/MetricView';
import { RepresentationView } from './components/RepresentationView';
import { LayerView } from './components/LayerView';
import { NeuronView } from './components/NeuronView';
import { InstanceView } from './components/InstanceView';
import { ActivationProjectionView } from './components/ActivationProjectionView';

// 导航项配置
const navItems = [
  { id: 'dashboard', label: 'Dashboard', icon: <GridViewIcon /> },
  { id: 'metrics', label: 'Metrics', icon: <DashboardIcon /> },
  { id: 'representation', label: 'Representation', icon: <TimelineIcon /> },
  { id: 'activation-projection', label: 'Activation Projection', icon: <ShowChartIcon /> },
  { id: 'layers', label: 'Layers', icon: <LayersIcon /> },
  { id: 'neurons', label: 'Neurons', icon: <HubIcon /> },
  { id: 'instances', label: 'Instances', icon: <ListAltIcon /> },
];

// Dashboard布局组件
const DashboardLayout: React.FC = () => {
  return (
    <Box sx={{ display: 'grid', gridTemplateColumns: '280px 1fr 1fr', gridTemplateRows: 'auto 1fr 1fr', gap: 2, height: 'calc(100vh - 140px)' }}>
      {/* A: 控制面板 - 左上 */}
      <Box sx={{ gridColumn: '1', gridRow: '1 / 4', overflow: 'auto' }}>
        <ControlPanel />
      </Box>

      {/* B: 指标视图 - 中左 */}
      <Box sx={{ gridColumn: '2', gridRow: '1', minHeight: 350 }}>
        <MetricView />
      </Box>

      {/* D: 层级视图 - 中上 (右侧) */}
      <Box sx={{ gridColumn: '3', gridRow: '1', minHeight: 350 }}>
        <LayerView />
      </Box>

      {/* C: 表征视图 - 下左 */}
      <Box sx={{ gridColumn: '2', gridRow: '2', minHeight: 350 }}>
        <RepresentationView />
      </Box>

      {/* E: 神经元视图 - 右中 */}
      <Box sx={{ gridColumn: '3', gridRow: '2', minHeight: 350 }}>
        <NeuronView />
      </Box>

      {/* F: 实例视图 - 底部中间 */}
      <Box sx={{ gridColumn: '2 / 4', gridRow: '3', minHeight: 300, overflow: 'auto' }}>
        <InstanceView />
      </Box>
    </Box>
  );
};

const App: React.FC = () => {
  const {
    activeView,
    setActiveView,
    isLoading,
    error,
    setError,
  } = useStore();

  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);

  const handleMenuClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleNavClick = (viewId: string) => {
    setActiveView(viewId as typeof activeView);
    handleMenuClose();
  };

  // 渲染单个视图
  const renderView = () => {
    switch (activeView) {
      case 'dashboard':
        return <DashboardLayout />;
      case 'metrics':
        return <MetricView />;
      case 'representation':
        return <RepresentationView />;
      case 'activation-projection':
        return <ActivationProjectionView />;
      case 'layers':
        return <LayerView />;
      case 'neurons':
        return <NeuronView />;
      case 'instances':
        return <InstanceView />;
      default:
        return <DashboardLayout />;
    }
  };

  const currentNavItem = navItems.find(item => item.id === activeView) || navItems[0];

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', bgcolor: '#f0f4f8' }}>
      {/* Top Navigation Bar */}
      <AppBar
        position="fixed"
        sx={{
          zIndex: (theme) => theme.zIndex.drawer + 1,
          background: 'linear-gradient(90deg, #0d1117 0%, #161b22 50%, #0d1117 100%)',
          borderBottom: '1px solid #30363d',
          boxShadow: 'none',
        }}
      >
        <Toolbar>
          {/* Menu Button */}
          <IconButton
            edge="start"
            color="inherit"
            aria-label="menu"
            onClick={handleMenuClick}
            sx={{ mr: 2 }}
          >
            <MenuIcon />
          </IconButton>

          {/* Logo and Title */}
          <Typography 
            variant="h5" 
            noWrap 
            component="div" 
            sx={{ 
              fontWeight: 700, 
              background: 'linear-gradient(135deg, #58a6ff 0%, #a371f7 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              letterSpacing: '0.5px'
            }}
          >
            🧠 NeuroLens
          </Typography>

          {/* Current View Indicator */}
          <Box sx={{ 
            ml: 4, 
            px: 2, 
            py: 0.5, 
            borderRadius: 1, 
            bgcolor: 'rgba(88, 166, 255, 0.1)',
            border: '1px solid rgba(88, 166, 255, 0.3)'
          }}>
            <Typography variant="body2" sx={{ color: '#58a6ff', fontWeight: 500 }}>
              {currentNavItem.label}
            </Typography>
          </Box>

          <Box sx={{ flexGrow: 1 }} />

          {/* Status */}
          <Typography variant="caption" sx={{ color: '#8b949e', mr: 2 }}>
            Jailbreak Neuron Analysis
          </Typography>
        </Toolbar>
      </AppBar>

      {/* Navigation Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
        PaperProps={{
          sx: {
            bgcolor: '#161b22',
            color: '#c9d1d9',
            border: '1px solid #30363d',
            mt: 1,
          }
        }}
      >
        {navItems.map((item) => (
          <MenuItem 
            key={item.id}
            onClick={() => handleNavClick(item.id)}
            selected={activeView === item.id}
            sx={{
              '&.Mui-selected': {
                bgcolor: 'rgba(88, 166, 255, 0.15)',
                borderLeft: '3px solid #58a6ff',
              },
              '&:hover': {
                bgcolor: 'rgba(88, 166, 255, 0.08)',
              },
              minWidth: 200,
            }}
          >
            <ListItemIcon sx={{ color: activeView === item.id ? '#1976d2' : '#666', minWidth: 36 }}>
              {item.icon}
            </ListItemIcon>
            <ListItemText 
              primary={item.label}
              primaryTypographyProps={{ 
                fontSize: '0.9rem',
                fontWeight: activeView === item.id ? 600 : 400
              }} 
            />
          </MenuItem>
        ))}
      </Menu>

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 2,
          mt: 8,
          bgcolor: '#f0f4f8',
          minHeight: 'calc(100vh - 64px)',
          overflow: 'auto',
        }}
      >
        {/* Error Alert */}
        {error && (
          <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {/* Loading */}
        {isLoading ? (
          <Box
            sx={{
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              minHeight: 400,
            }}
          >
            <CircularProgress sx={{ color: '#58a6ff' }} />
          </Box>
        ) : (
          /* View Content */
          <Box
            sx={{
              backgroundColor: 'rgba(22, 27, 34, 0.8)',
              borderRadius: 2,
              border: '1px solid #30363d',
              p: activeView === 'dashboard' ? 1 : 3,
              minHeight: activeView === 'dashboard' ? 'calc(100vh - 100px)' : 600,
            }}
          >
            {renderView()}
          </Box>
        )}
      </Box>
    </Box>
  );
};

export default App;
