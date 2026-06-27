import React from 'react';

const WeatherIntelPanel = ({ intelData }) => {
  if (!intelData) return null;

  return (
    <div className="mt-8 bg-gray-900 p-6 rounded-lg border border-gray-700 text-white">
      <h2 className="text-2xl font-bold mb-4 text-cyan-400">🌩️ Operational Weather Intelligence</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-gray-800 p-4 rounded">
          <p className="text-gray-400 text-sm">Cloud Coverage</p>
          <p className="text-xl font-bold">{intelData.cloud_coverage_pct}%</p>
        </div>
        <div className="bg-gray-800 p-4 rounded">
          <p className="text-gray-400 text-sm">Motion Speed</p>
          <p className="text-xl font-bold">{intelData.cloud_motion_speed_px} px/s</p>
        </div>
        <div className="bg-gray-800 p-4 rounded border-l-4 border-red-500">
          <p className="text-gray-400 text-sm">System Alert Level</p>
          <p className="text-lg font-bold text-red-400">{intelData.weather_alert_level}</p>
        </div>
      </div>

      <h3 className="text-xl font-semibold mb-3">Spatial Analysis (Physics Constraints)</h3>
      <div className="flex flex-col md:flex-row gap-6">
        <div className="flex-1">
          <p className="text-sm text-gray-400 mb-2">Absolute Spatial Error Heatmap</p>
          <img 
            src="http://localhost:8080/api/visuals/heatmap" 
            alt="Error Heatmap" 
            className="w-full rounded border border-gray-600"
            onError={(e) => e.target.style.display = 'none'}
          />
        </div>
        <div className="flex-1">
          <p className="text-sm text-gray-400 mb-2">Cloud Motion Vector Field</p>
          <img 
            src="http://localhost:8080/api/visuals/vectors" 
            alt="Motion Vectors" 
            className="w-full rounded border border-gray-600"
            onError={(e) => e.target.style.display = 'none'}
          />
        </div>
      </div>
    </div>
  );
};

export default WeatherIntelPanel;
