import { useMcp } from 'use-mcp/react';

const getMcpServerUrl = () => {
  // Check localStorage first, then environment variable, then default
  return localStorage.getItem('mcpServerUrl') ||
         import.meta.env.VITE_MCP_SERVER_URL ||
         'http://localhost:8000/mcp';
};

export function useRagMcp() {
  const mcpServerUrl = getMcpServerUrl();

  const {
    state,
    error,
    tools,
    callTool,
    retry,
    authenticate,
    clearStorage,
  } = useMcp({
    url: mcpServerUrl,
    clientName: 'JAC Safety Regulations Search',
    autoReconnect: true,
    autoRetry: 3000,
    transportType: 'http', // Use HTTP streaming (modern MCP protocol)
    debug: true,
  });

  const searchDocuments = async (query, maxResults = 10, searchMode = 'hybrid') => {
    if (state !== 'ready') {
      throw new Error('MCP server not ready');
    }

    // Call the search tool on the MCP server
    const result = await callTool('search_documents', {
      query,
      max_results: maxResults,
      search_mode: searchMode,
    });

    // MCP returns data in structuredContent or parse from content[0].text
    if (result.structuredContent) {
      return result.structuredContent;
    } else if (result.content && result.content[0]?.text) {
      return JSON.parse(result.content[0].text);
    }
    return result;
  };

  const listDocuments = async () => {
    if (state !== 'ready') {
      throw new Error('MCP server not ready');
    }

    // Call the list_documents tool on the MCP server
    const result = await callTool('list_documents', {});

    // MCP returns data in structuredContent or parse from content[0].text
    if (result.structuredContent) {
      return result.structuredContent;
    } else if (result.content && result.content[0]?.text) {
      return JSON.parse(result.content[0].text);
    }
    return result;
  };

  return {
    state,
    error,
    tools,
    searchDocuments,
    listDocuments,
    retry,
    authenticate,
    clearStorage,
  };
}
