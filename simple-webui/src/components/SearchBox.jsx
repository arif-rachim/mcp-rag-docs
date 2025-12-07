import { useState } from 'react';

function SearchBox({ onSearch, loading, disabled }) {
  const [query, setQuery] = useState('');
  const [maxResults, setMaxResults] = useState(10);
  const [searchMode, setSearchMode] = useState('hybrid');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query, maxResults, searchMode);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="mb-6">
      <div className="flex items-center border border-gray-300 rounded-full px-5 py-3 hover:shadow-md focus-within:shadow-md transition-shadow">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search JAC safety regulations..."
          className="flex-1 outline-none text-base"
          disabled={loading || disabled}
        />
        <div className="flex items-center space-x-3 ml-4">
          <select
            value={searchMode}
            onChange={(e) => setSearchMode(e.target.value)}
            className="px-2 py-1 border border-gray-300 rounded text-sm outline-none bg-white"
            disabled={loading || disabled}
            title="Search mode"
          >
            <option value="hybrid">Hybrid</option>
            <option value="semantic">Semantic</option>
            <option value="keyword">Keyword</option>
          </select>
          <input
            type="number"
            value={maxResults}
            onChange={(e) => setMaxResults(parseInt(e.target.value))}
            min="1"
            max="25"
            className="w-12 px-2 py-1 border border-gray-300 rounded text-sm text-center outline-none"
            disabled={loading || disabled}
            title="Max results"
          />
          <button
            type="submit"
            disabled={loading || disabled || !query.trim()}
            className="text-gray-600 hover:text-black disabled:text-gray-300"
          >
            {loading ? (
              <svg className="w-5 h-5 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
            ) : (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            )}
          </button>
        </div>
      </div>
    </form>
  );
}

export default SearchBox;
