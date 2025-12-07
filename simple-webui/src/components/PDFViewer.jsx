import { useState } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';

// Configure PDF.js worker - use local worker file from public folder
pdfjs.GlobalWorkerOptions.workerSrc = '/pdf.worker.min.mjs';

function PDFViewer({ filename, pageNumber, onClose }) {
  const [numPages, setNumPages] = useState(null);
  const [currentPage, setCurrentPage] = useState(pageNumber || 1);
  const [scale, setScale] = useState(1.0);

  const pdfUrl = `http://localhost:8000/pdfs/${encodeURIComponent(filename)}`;

  function onDocumentLoadSuccess({ numPages }) {
    setNumPages(numPages);
    // Auto-navigate to specified page
    if (pageNumber && pageNumber <= numPages) {
      setCurrentPage(pageNumber);
    }
  }

  return (
    <div className="fixed right-0 top-0 h-screen w-1/2 bg-white border-l border-gray-300 flex flex-col z-40">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-300">
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-medium truncate">{filename}</h3>
          <p className="text-xs text-gray-500">
            Page {currentPage} of {numPages || '?'}
          </p>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2 ml-4">
          {/* Zoom controls */}
          <button
            onClick={() => setScale(s => Math.max(0.5, s - 0.1))}
            className="px-2 py-1 border border-gray-300 text-sm hover:bg-gray-50"
            title="Zoom out"
          >
            −
          </button>
          <span className="text-xs text-gray-600 min-w-[3rem] text-center">
            {Math.round(scale * 100)}%
          </span>
          <button
            onClick={() => setScale(s => Math.min(2.0, s + 0.1))}
            className="px-2 py-1 border border-gray-300 text-sm hover:bg-gray-50"
            title="Zoom in"
          >
            +
          </button>

          {/* Close button */}
          <button
            onClick={onClose}
            className="ml-4 px-3 py-1 bg-gray-100 hover:bg-gray-200 text-sm"
          >
            Close
          </button>
        </div>
      </div>

      {/* Page navigation */}
      {numPages && (
        <div className="flex items-center justify-center gap-3 px-4 py-2 border-b border-gray-200 bg-gray-50">
          <button
            onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
            disabled={currentPage <= 1}
            className="px-3 py-1 border border-gray-300 text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 bg-white"
          >
            Previous
          </button>
          <span className="text-sm flex items-center gap-2">
            Page
            <input
              type="number"
              value={currentPage}
              onChange={(e) => {
                const page = parseInt(e.target.value);
                if (page >= 1 && page <= numPages) {
                  setCurrentPage(page);
                }
              }}
              className="w-16 px-2 py-1 border border-gray-300 text-center"
              min="1"
              max={numPages}
            />
            of {numPages}
          </span>
          <button
            onClick={() => setCurrentPage(p => Math.min(numPages, p + 1))}
            disabled={currentPage >= numPages}
            className="px-3 py-1 border border-gray-300 text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 bg-white"
          >
            Next
          </button>
        </div>
      )}

      {/* PDF viewer */}
      <div className="flex-1 overflow-auto bg-gray-100 p-4">
        <div className="flex justify-center">
          <Document
            file={pdfUrl}
            onLoadSuccess={onDocumentLoadSuccess}
            loading={
              <div className="text-center py-8">
                <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
                <p className="mt-2 text-gray-600">Loading PDF...</p>
              </div>
            }
            error={
              <div className="text-center py-8 text-red-600">
                <p className="font-medium">Failed to load PDF</p>
                <p className="text-sm mt-1">File: {filename}</p>
              </div>
            }
          >
            <Page
              pageNumber={currentPage}
              scale={scale}
              renderTextLayer={true}
              renderAnnotationLayer={true}
              className="shadow-lg"
            />
          </Document>
        </div>
      </div>
    </div>
  );
}

export default PDFViewer;
