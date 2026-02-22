// NeuroLens Visualization - Main App Component

import React from 'react';
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Container,
  CircularProgress,
  Alert,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Timeline as TimelineIcon,
  Layers as LayersIcon,
  Hub as HubIcon,
  ListAlt as ListAltIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';

import { useStore } from './store';
import { ControlPanel } from './components/ControlPanel';
import { MetricView } from './components/MetricView';
import { RepresentationView } from './components/RepresentationView';
import { LayerView } from './components/LayerView';
import { NeuronView } from './components/NeuronView';
import { InstanceView } from './components/InstanceView';

const drawerWidth = 240;

const navItems = [
  { id: 'metrics', label: 'Metric View', icon: <DashboardIcon /> },
  { id: 'representation', label: 'Representation View', icon: <TimelineIcon /> },
  { id: 'layers', label: 'Layer View', icon: <LayersIcon /> },
  { id: 'neurons', label: 'Neuron View', icon: <HubIcon /> },
  { id: 'instances', label: 'Instance View', icon: <ListAltIcon /> },
];

const App: React.FC = () => {
  const {
    activeView,
    setActiveView,
    isLoading,
    error,
    setError,
  } = useStore();

  const renderView = () => {
    switch (activeView) {
      case 'metrics':
        return <MetricView />;
      case 'representation':
        return <RepresentationView />;
      case 'layers':
        return <LayerView />;
      case 'neurons':
        return <NeuronView />;
      case 'instances':
        return <InstanceView />;
      default:
        return <MetricView />;
    }
  };

  return (
    <Box sx={{ display: 'flex' }}>
      {/* App Bar */}
      <AppBar
        position="fixed"
        sx={{
          zIndex: (theme) => theme.zIndex.drawer + 1,
          background: 'linear-gradient(135deg, #1a237e 0%, #3949ab 100%)',
        }}
      >
        <Toolbar>
          <Typography variant="h6" noWrap component="div">
            NeuroLens Visualization
          </Typography>
          <Box sx={{ flexGrow: 1 }} />
          <Typography variant="body2" sx={{ opacity: 0.8 }}>
            v1.0.0
          </Typography>
        </Toolbar>
      </AppBar>

      {/* Drawer Sidebar */}
      <Drawer
        variant="permanent"
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            boxSizing: 'border-box',
            marginTop: '64px',
          },
        }}
      >
        <Box sx={{ overflow: 'auto', mt: 2 }}>
          <List>
            {navItems.map((item) => (
              <ListItem
                button
                key={item.id}
                onClick={() => setActiveView(item.id as typeof activeView)}
                selected={activeView === item.id}
                sx={{
                  '&.Mui-selected': {
                    backgroundColor: 'rgba(25, 118, 210, 0.08)',
                    borderRight: '3px solid #1976d2',
                  },
                  '&:hover': {
                    backgroundColor: 'rgba(25, 118, 210, 0.04)',
                  },
                }}
              >
                <ListItemIcon sx={{ color: activeView === item.id ? '#1976d2' : 'inherit' }}>
                  {item.icon}
                </ListItemIcon>
                <ListItemText primary={item.label} />
              </ListItem>
            ))}
          </List>
        </Box>
      </Drawer>

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          mt: 8,
          backgroundColor: '#f5f5f5',
          minHeight: '100vh',
        }}
      >
        {/* Control Panel */}
        <Box sx={{ mb: 3 }}>
          <ControlPanel />
        </Box>

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
            <CircularProgress />
          </Box>
        ) : (
          /* View Content */
          <Box
            sx={{
              backgroundColor: 'white',
              borderRadius: 2,
              boxShadow: 1,
              p: 3,
              minHeight: 600,
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

