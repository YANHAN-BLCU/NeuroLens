// Activation Projection View Component

import React from 'react';
import { Box, Typography, Paper } from '@mui/material';

export const ActivationProjectionView: React.FC = () => {
  return (
    <Box>
      <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
        Activation Projection View
      </Typography>
      <Paper sx={{ p: 4, textAlign: 'center' }}>
        <Typography variant="body1" color="text.secondary">
          Activation projection data not yet available. Please run the activation projection pipeline first.
        </Typography>
      </Paper>
    </Box>
  );
};
