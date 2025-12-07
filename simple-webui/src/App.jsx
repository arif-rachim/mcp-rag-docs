import { useState } from 'react';
import { useRagMcp } from './hooks/useRagMcp';
import SearchBox from './components/SearchBox';
import ResultsView from './components/ResultsView';
import DocumentList from './components/DocumentList';
import ConnectionStatus from './components/ConnectionStatus';
import Settings from './components/Settings';
import PDFViewer from './components/PDFViewer';

function App() {
  const { state, error, tools, searchDocuments, listDocuments, retry, authenticate } = useRagMcp();
  const [activeTab, setActiveTab] = useState('search');
  const [searchResults, setSearchResults] = useState(null);
  const [documents, setDocuments] = useState(null);
  const [loading, setLoading] = useState(false);
  const [opError, setOpError] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const [selectedPdf, setSelectedPdf] = useState(null);

  const handleSearch = async (query, maxResults, searchMode) => {
    setLoading(true);
    setOpError(null);
    setSelectedPdf(null); // Close PDF when new search
    try {
      const results = await searchDocuments(query, maxResults, searchMode);
      setSearchResults(results);
    } catch (err) {
      setOpError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleResultClick = (result) => {
    setSelectedPdf({
      filename: result.metadata.filename,
      pageNumber: result.metadata.page,
    });
  };

  const handleLoadDocuments = async () => {
    setLoading(true);
    setOpError(null);
    try {
      const docs = await listDocuments();
      setDocuments(docs);
    } catch (err) {
      setOpError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-white">
      <header className="border-b border-gray-200 py-3">
        <div className="max-w-[1400px] mx-auto px-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-8">
              <h1 className="text-xl font-normal text-black">JAC Safety Regulations</h1>
              <nav className="flex space-x-4 text-sm">
                <button
                  onClick={() => setActiveTab('search')}
                  className={`pb-3 border-b-2 transition-colors ${
                    activeTab === 'search'
                      ? 'border-black text-black'
                      : 'border-transparent text-gray-600 hover:text-black'
                  }`}
                  disabled={state !== 'ready'}
                >
                  Search
                </button>
                <button
                  onClick={() => { setActiveTab('documents'); handleLoadDocuments(); }}
                  className={`pb-3 border-b-2 transition-colors ${
                    activeTab === 'documents'
                      ? 'border-black text-black'
                      : 'border-transparent text-gray-600 hover:text-black'
                  }`}
                  disabled={state !== 'ready'}
                >
                  Documents
                </button>
              </nav>
            </div>
            <button
              onClick={() => setShowSettings(true)}
              className="text-gray-600 hover:text-black transition-colors text-sm"
              title="Settings"
            >
              Settings
            </button>
          </div>
        </div>
      </header>

      <ConnectionStatus state={state} error={error} onRetry={retry} onAuthenticate={authenticate} />

      <main
        className={`mx-auto px-8 py-6 transition-all duration-300 ${
          selectedPdf ? 'max-w-[50%] mr-[50%]' : 'max-w-[1400px]'
        }`}
      >
        {opError && (
          <div className="border border-gray-300 bg-gray-50 px-4 py-3 mb-4 text-sm">
            {opError}
          </div>
        )}

        {activeTab === 'search' && (
          <>
            <SearchBox onSearch={handleSearch} loading={loading} disabled={state !== 'ready'} />
            <ResultsView
              results={searchResults}
              loading={loading}
              onResultClick={handleResultClick}
            />
          </>
        )}

        {activeTab === 'documents' && (
          <DocumentList documents={documents} loading={loading} />
        )}
      </main>

      {selectedPdf && (
        <PDFViewer
          filename={selectedPdf.filename}
          pageNumber={selectedPdf.pageNumber}
          onClose={() => setSelectedPdf(null)}
        />
      )}

      {showSettings && <Settings onClose={() => setShowSettings(false)} />}
    </div>
  );
}

export default App;
