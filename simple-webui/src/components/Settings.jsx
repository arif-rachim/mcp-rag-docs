import { useState } from 'react';

function Settings({ onClose }) {
  const [mcpUrl, setMcpUrl] = useState(
    localStorage.getItem('mcpServerUrl') || import.meta.env.VITE_MCP_SERVER_URL || 'http://localhost:8000/mcp'
  );
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    localStorage.setItem('mcpServerUrl', mcpUrl);
    setSaved(true);
    setTimeout(() => {
      setSaved(false);
      window.location.reload(); // Reload to apply changes
    }, 1500);
  };

  const handleReset = () => {
    const defaultUrl = import.meta.env.VITE_MCP_SERVER_URL || 'http://localhost:8000/mcp';
    setMcpUrl(defaultUrl);
    localStorage.removeItem('mcpServerUrl');
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-20 flex items-center justify-center z-50">
      <div className="bg-white border border-gray-300 p-8 max-w-md w-full mx-4">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-light">Settings</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-black text-2xl leading-none font-light"
          >
            ×
          </button>
        </div>

        <div className="space-y-6">
          <div>
            <label className="block text-sm font-light text-gray-600 mb-2">
              MCP Server URL (HTTP)
            </label>
            <input
              type="text"
              value={mcpUrl}
              onChange={(e) => setMcpUrl(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 focus:border-black focus:outline-none font-light"
              placeholder="http://localhost:8000/mcp"
            />
            <p className="mt-2 text-xs font-light text-gray-400">
              The HTTP endpoint URL for the MCP server (modern streaming protocol)
            </p>
          </div>

          {saved && (
            <div className="bg-gray-50 border border-gray-300 text-black px-4 py-3 font-light text-sm">
              Settings saved! Reloading...
            </div>
          )}

          <div className="flex gap-3 pt-4">
            <button
              onClick={handleSave}
              className="flex-1 bg-black text-white px-4 py-2 hover:bg-gray-800 transition-colors font-light"
            >
              Save & Reload
            </button>
            <button
              onClick={handleReset}
              className="px-4 py-2 border border-gray-300 hover:border-black transition-colors font-light"
            >
              Reset
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Settings;
