import { useState, useEffect, useRef } from 'react';

export default function VideoPlayer({ videoUrl, products, onProductClick, onTimeUpdate }) {
  const [currentTime, setCurrentTime] = useState(0);
  const [visibleProducts, setVisibleProducts] = useState([]);
  const videoRef = useRef(null);

  // Update visible products based on current time
  useEffect(() => {
    const visible = products.filter(p =>
      currentTime >= p.timeline[0] && currentTime <= p.timeline[1]
    );
    setVisibleProducts(visible);
  }, [currentTime, products]);

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      const time = videoRef.current.currentTime;
      setCurrentTime(time);
      if (onTimeUpdate) {
        onTimeUpdate(time);
      }
    }
  };

  const getMarkerPosition = (product) => {
    const [x, y] = product.location;
    return { left: `${x}%`, top: `${y}%` };
  };

  return (
    <div className="relative bg-black rounded-lg overflow-hidden" style={{ paddingTop: '56.25%' }}>
      <div className="absolute inset-0">
        <video
          ref={videoRef}
          src={videoUrl}
          controls
          autoPlay
          muted
          playsInline
          onTimeUpdate={handleTimeUpdate}
          onError={(e) => console.error('Video error:', e)}
          className="w-full h-full"
          style={{ display: 'block', width: '100%', height: '100%' }}
        />

        {/* Product markers */}
        <div className="absolute inset-0 pointer-events-none">
          {visibleProducts.map((product, idx) => {
            const pos = getMarkerPosition(product);
            return (
              <button
                key={idx}
                className="absolute w-12 h-12 bg-black bg-opacity-70 rounded-full flex items-center justify-center pointer-events-auto hover:bg-opacity-90 transition-all animate-pulse"
                style={{ left: pos.left, top: pos.top, transform: 'translate(-50%, -50%)' }}
                onClick={() => onProductClick(product)}
                aria-label={`View ${product.product_name}`}
              >
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 11V7a4 4 0 00-8 0v4M5 9h14l1 12H4L5 9z" />
                </svg>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
