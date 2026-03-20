import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './VideoPlayer.css';
import LiveFeedLabel from './LiveFeedLabel';
import { API_URL } from '../config';

const VideoPlayer = ({ hasSource, activeConfigId }) => {
  const [inferenceReady, setInferenceReady] = useState(false);

  useEffect(() => {
    if (!hasSource) {
      setInferenceReady(false);
      return;
    }
    setInferenceReady(false); /* reset when source/config changes */
    const interval = setInterval(async () => {
      try {
        const res = await axios.get(`${API_URL}/latest_info`);
        const ready = res.data.inference_ready ?? false;
        setInferenceReady(ready);
        if (ready) {
          clearInterval(interval);
        }
      } catch (err) {
        setInferenceReady(false);
      }
    }, 500);
    return () => clearInterval(interval);
  }, [hasSource, activeConfigId]);

  const feedUrl = `${API_URL}/video_feed${activeConfigId != null ? `?config=${activeConfigId}` : ''}`;
  return (
    <div className="video-feed-container">
      {hasSource ? (
        <>
          <LiveFeedLabel />
          <img src={feedUrl} alt="Video Feed" key={activeConfigId} />
          {!inferenceReady && (
            <div className="video-loading-overlay" aria-live="polite">
              Loading model...
            </div>
          )}
        </>
      ) : (
        <div className="video-feed-placeholder" aria-live="polite">
          Select a source to start
        </div>
      )}
    </div>
  );
};

export default VideoPlayer;
