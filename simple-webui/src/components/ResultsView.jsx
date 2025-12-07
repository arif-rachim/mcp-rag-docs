function ResultsView({ results, loading, onResultClick }) {
  if (loading) {
    return (
      <div className="space-y-5">
        {[1, 2, 3].map((i) => (
          <div key={i} className="animate-pulse">
            <div className="flex items-start">
              <div className="flex-shrink-0 w-10 h-10 bg-gray-200 rounded-full mr-3"></div>
              <div className="flex-1 min-w-0 space-y-2">
                <div className="h-3 bg-gray-200 rounded w-1/4"></div>
                <div className="h-5 bg-gray-200 rounded w-3/4"></div>
                <div className="h-4 bg-gray-200 rounded w-full"></div>
                <div className="h-4 bg-gray-200 rounded w-5/6"></div>
                <div className="h-3 bg-gray-200 rounded w-1/3"></div>
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (!results || !results.results) return null;

  const { results: items, total_found, query } = results;

  return (
    <div>
      <div className="mb-4">
        <p className="text-sm text-gray-600">
          About {total_found} results
        </p>
      </div>

      <div className="space-y-5">
        {items.map((item, idx) => (
          <div key={idx} className="group">
            <div className="flex items-start">
              <div className="flex-shrink-0 w-10 h-10 bg-gray-100 rounded-full flex items-center justify-center text-xs text-gray-600 mr-3">
                {item.metadata.page}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-baseline justify-between mb-1">
                  <div className="flex items-center text-xs text-gray-600 space-x-2">
                    <span className="truncate">{item.metadata.filename}</span>
                    {item.metadata.jac_reg && (
                      <span className="text-black font-medium">{item.metadata.jac_reg}</span>
                    )}
                    {item.metadata.jac_sgl && (
                      <span className="text-black font-medium">{item.metadata.jac_sgl}</span>
                    )}
                  </div>
                </div>

                <h3
                  onClick={() => onResultClick && onResultClick(item)}
                  className="text-xl text-blue-600 hover:underline cursor-pointer mb-1 hover:text-blue-800 transition-colors"
                >
                  {item.metadata.filename} - Page {item.metadata.page}
                </h3>

                <div className="text-sm text-gray-600 leading-relaxed line-clamp-3">
                  {item.text}
                </div>

                <div className="flex items-center text-xs text-gray-500 mt-2 space-x-2">
                  <span>{item.metadata.lang}</span>
                  <span>•</span>
                  <span className="font-medium">
                    {item.rerank_score ? (
                      <>
                        Vector: <span className="text-gray-600">{item.score?.toFixed(3)}</span>
                        <span className="mx-1">•</span>
                        Rerank: <span className="text-blue-600">{item.rerank_score.toFixed(3)}</span>
                      </>
                    ) : (
                      <>Score: {item.score?.toFixed(3)}</>
                    )}
                  </span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default ResultsView;
