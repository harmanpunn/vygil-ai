import React, { useEffect, useRef } from 'react';

export default function ProductSidebar({ products, currentTime, onProductClick, isLoading }) {
  const productRefs = useRef([]);

  // Auto-scroll to active product - must be declared before any returns
  useEffect(() => {
    if (!products || products.length === 0) return;

    const activeIndex = products.findIndex(
      p => currentTime >= p.timeline[0] && currentTime <= p.timeline[1]
    );

    if (activeIndex !== -1 && productRefs.current[activeIndex]) {
      productRefs.current[activeIndex].scrollIntoView({
        behavior: 'smooth',
        block: 'center'
      });
    }
  }, [currentTime, products]);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6 h-full">
        <h2 className="text-xl font-semibold mb-4">Products</h2>
        <div className="flex items-center gap-2">
          <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
          <p className="text-gray-600">Analyzing video...</p>
        </div>
      </div>
    );
  }

  if (!products.length) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6 h-full text-center">
        <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
          <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 11V7a4 4 0 00-8 0v4M5 9h14l1 12H4L5 9z" />
          </svg>
        </div>
        <h2 className="text-xl font-semibold mb-2">No Products Found</h2>
        <p className="text-gray-600">This video doesn't contain detectable products</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md h-full flex flex-col">
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-xl font-semibold">Products ({products.length})</h2>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
        {products.map((product, idx) => {
          const isActive = currentTime >= product.timeline[0] && currentTime <= product.timeline[1];

          return (
            <div
              key={idx}
              ref={(el) => (productRefs.current[idx] = el)}
              className={`p-4 rounded-lg border-2 transition-all cursor-pointer ${
                isActive
                  ? 'border-blue-500 bg-blue-50 shadow-lg'
                  : 'border-gray-200 bg-white hover:border-gray-300'
              }`}
              onClick={() => onProductClick(product)}
            >
              <div className="flex items-center gap-2 mb-2">
                <span className={`text-xs px-2 py-1 rounded ${
                  isActive ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-600'
                }`}>
                  {formatTime(product.timeline[0])} - {formatTime(product.timeline[1])}
                </span>
              </div>

              <h3 className={`font-semibold mb-1 ${isActive ? 'text-gray-900' : 'text-gray-500'}`}>
                {product.product_name}
              </h3>

              {product.brand && product.brand.toLowerCase() !== 'unknown' && (
                <p className={`text-sm mb-1 ${isActive ? 'text-gray-700' : 'text-gray-400'}`}>
                  Brand: {product.brand}
                </p>
              )}

              {product.price && product.price.toLowerCase() !== 'unknown' && product.price.toLowerCase() !== 'not shown in the video' && (
                <p className={`text-sm font-semibold mb-2 ${isActive ? 'text-gray-900' : 'text-gray-500'}`}>
                  {product.price}
                </p>
              )}

              <p className={`text-sm mb-3 ${isActive ? 'text-gray-600' : 'text-gray-400'}`}>
                {product.description}
              </p>

              <button
                className={`w-full py-2 px-4 rounded transition-colors ${
                  isActive
                    ? 'bg-black text-white hover:bg-gray-800'
                    : 'bg-gray-200 text-gray-500 cursor-not-allowed'
                }`}
                disabled={!isActive}
                onClick={(e) => {
                  e.stopPropagation();
                  if (isActive) {
                    const brand = product.brand && product.brand.toLowerCase() !== 'unknown' ? product.brand : '';
                    const query = brand ? `${brand} ${product.product_name}` : product.product_name;
                    window.open(`https://www.amazon.com/s?k=${encodeURIComponent(query)}`, '_blank');
                  }
                }}
              >
                Shop on Amazon
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
