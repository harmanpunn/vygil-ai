import React, { useState, useEffect, useRef } from 'react';
import VideoPlayer from './VideoPlayer';
import ProductSidebar from './ProductSidebar';

export default function VideoIntelligence() {
  const [videos, setVideos] = useState([]);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [products, setProducts] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const videoContainerRef = useRef(null);
  const [sidebarHeight, setSidebarHeight] = useState(0);

  // Load videos on mount
  useEffect(() => {
    loadVideos();
  }, []);

  // Calculate sidebar height to match video player
  useEffect(() => {
    const updateSidebarHeight = () => {
      if (videoContainerRef.current) {
        const height = videoContainerRef.current.offsetHeight;
        setSidebarHeight(height);
      }
    };

    updateSidebarHeight();
    window.addEventListener('resize', updateSidebarHeight);

    return () => window.removeEventListener('resize', updateSidebarHeight);
  }, [selectedVideo]);

  const loadVideos = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/videos');
      const data = await response.json();
      if (data.success && data.data.length > 0) {
        setVideos(data.data);
        // Auto-select first video that has HLS URL
        const videoWithHls = data.data.find(v => v.hls?.video_url);
        if (videoWithHls) {
          loadVideo(videoWithHls._id);
        } else {
          // Fallback to first video if none have HLS
          loadVideo(data.data[0]._id);
        }
      }
    } catch (error) {
      console.error('Failed to load videos:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const loadVideo = async (videoId) => {
    setIsAnalyzing(true);
    setProducts([]);
    setCurrentTime(0);

    try {
      // Get video details
      const videoResponse = await fetch(`/api/videos/${videoId}`);
      const videoData = await videoResponse.json();
      setSelectedVideo(videoData);

      // Get/analyze products
      const productsResponse = await fetch(`/api/videos/${videoId}/products`);
      const productsData = await productsResponse.json();

      if (productsData.success) {
        setProducts(productsData.products || []);
      }
    } catch (error) {
      console.error('Failed to load video:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleProductClick = (product) => {
    // Could add highlight effect or auto-scroll here if needed
    console.log('Product clicked:', product);
  };

  const getVideoName = (video) => {
    const name = video.system_metadata?.filename ||
           video.system_metadata?.video_title ||
           `Video ${video._id.slice(-8)}`;

    // Add indicator if video doesn't have HLS
    if (!video.hls?.video_url) {
      return `${name} (No stream available)`;
    }
    return name;
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600">Loading videos...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4 max-h-[calc(100vh-140px)] overflow-hidden">
      {/* Header and Info in one row */}
      <div className="flex items-start justify-between gap-6">
        <div className="flex-1">
          <h1 className="text-2xl font-bold mb-2">Video Intelligence</h1>
          <p className="text-gray-600 text-sm">AI-powered product detection in your videos</p>
        </div>

        {/* Compact Info */}
        <div className="bg-gray-100 rounded-lg p-3 text-xs text-gray-700 max-w-xs">
          <p className="font-semibold mb-1">💡 How it works</p>
          <p>Select video → AI detects products → Click to shop on Amazon</p>
        </div>
      </div>

      {/* Video Selector */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Select Video</label>
        <select
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          value={selectedVideo?._id || ''}
          onChange={(e) => loadVideo(e.target.value)}
        >
          {videos.map((video) => (
            <option key={video._id} value={video._id}>
              {getVideoName(video)}
            </option>
          ))}
        </select>
      </div>

      {/* Main Content */}
      {selectedVideo && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
          {/* Video Player (2/3 width) */}
          <div className="lg:col-span-2" ref={videoContainerRef}>
            {selectedVideo.local_url || selectedVideo.hls?.video_url ? (
              <VideoPlayer
                videoUrl={selectedVideo.local_url || selectedVideo.hls.video_url}
                products={products}
                onProductClick={handleProductClick}
                onTimeUpdate={setCurrentTime}
              />
            ) : (
              <div className="bg-gray-100 rounded-lg p-8 text-center">
                <p className="text-gray-600">Video not available</p>
              </div>
            )}
          </div>

          {/* Product Sidebar (1/3 width) - Match video player height */}
          <div style={{ height: sidebarHeight > 0 ? `${sidebarHeight}px` : 'auto' }}>
            <ProductSidebar
              products={products}
              currentTime={currentTime}
              onProductClick={handleProductClick}
              isLoading={isAnalyzing}
            />
          </div>
        </div>
      )}
    </div>
  );
}
