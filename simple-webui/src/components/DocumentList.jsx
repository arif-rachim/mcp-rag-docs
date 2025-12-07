function DocumentList({ documents, loading }) {
  if (loading) {
    return <div className="text-center py-8 text-gray-500">Loading documents...</div>;
  }

  if (!documents) return null;

  const { documents: docs, summary } = documents;

  return (
    <div>
      <div className="mb-4">
        <p className="text-sm text-gray-600">
          {summary.total_documents} documents • {summary.total_chunks} chunks • {summary.total_pages} pages
        </p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr className="border-b border-gray-200">
              <th className="px-3 py-2 text-left text-xs text-gray-600">Filename</th>
              <th className="px-3 py-2 text-left text-xs text-gray-600">Pages</th>
              <th className="px-3 py-2 text-left text-xs text-gray-600">Chunks</th>
              <th className="px-3 py-2 text-left text-xs text-gray-600">Lang</th>
              <th className="px-3 py-2 text-left text-xs text-gray-600">JAC REG</th>
              <th className="px-3 py-2 text-left text-xs text-gray-600">JAC SGL</th>
            </tr>
          </thead>
          <tbody>
            {docs.map((doc, idx) => (
              <tr key={idx} className="border-b border-gray-100 hover:bg-gray-50">
                <td className="px-3 py-2 text-gray-900">{doc.filename}</td>
                <td className="px-3 py-2 text-gray-600">
                  {doc.indexed_pages}/{doc.total_pages}
                </td>
                <td className="px-3 py-2 text-gray-600">{doc.chunks}</td>
                <td className="px-3 py-2 text-gray-600">{doc.language}</td>
                <td className="px-3 py-2 text-gray-600">
                  {doc.jac_regulations?.join(', ') || '—'}
                </td>
                <td className="px-3 py-2 text-gray-600">
                  {doc.jac_guidelines?.join(', ') || '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default DocumentList;
