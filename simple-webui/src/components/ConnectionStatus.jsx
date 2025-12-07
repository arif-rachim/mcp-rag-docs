function ConnectionStatus({ state, error, onRetry, onAuthenticate }) {
  if (state === 'ready') return null;

  const statusMessages = {
    discovering: 'Discovering MCP server...',
    pending_auth: 'Authentication required',
    authenticating: 'Authenticating...',
    connecting: 'Connecting to MCP server...',
    loading: 'Loading tools...',
    failed: 'Connection failed',
  };

  const message = statusMessages[state] || 'Initializing...';
  const isError = state === 'failed' || state === 'pending_auth';

  return (
    <div className={`py-2 ${isError ? 'bg-red-50' : 'bg-gray-50'}`}>
      <div className="max-w-[1400px] mx-auto px-8">
        <div className="flex items-center justify-between">
          <p className="text-sm text-gray-700">
            {message}
            {error && <span className="text-red-600 ml-2">{error}</span>}
          </p>
          <div className="flex gap-2">
            {state === 'pending_auth' && (
              <button
                onClick={onAuthenticate}
                className="text-blue-600 hover:underline text-sm"
              >
                Authenticate
              </button>
            )}
            {state === 'failed' && (
              <button
                onClick={onRetry}
                className="text-blue-600 hover:underline text-sm"
              >
                Retry
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default ConnectionStatus;
