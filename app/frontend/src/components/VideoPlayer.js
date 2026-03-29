import React, { useState, useEffect, useLayoutEffect, useRef } from 'react';
import axios from 'axios';
import './VideoPlayer.css';
import LiveFeedLabel from './LiveFeedLabel';
import { API_URL } from '../config';

const VideoPlayer = ({ hasSource, activeConfigId }) => {
  const [inferenceReady, setInferenceReady] = useState(false);
  const [feedNonce, setFeedNonce] = useState(0);
  const imgRef = useRef(null);

  /* Before paint: show loading overlay and new stream URL immediately on switch */
  useLayoutEffect(() => {
    if (!hasSource) {
      setInferenceReady(false);
      return;
    }
    /* Abort prior multipart response: same <img> node + new src cancels the old fetch */
    if (imgRef.current) {
      imgRef.current.removeAttribute('src');
    }
    setInferenceReady(false);
    setFeedNonce((n) => n + 1);
  }, [hasSource, activeConfigId]);

  useEffect(() => {
    if (!hasSource) {
      return;
    }
    const interval = setInterval(async () => {
      try {
        const res = await axios.get(`${API_URL}/latest_info`, { params: { _: Date.now() } });
        const ready = Boolean(res.data.inference_ready);
        setInferenceReady(ready);
        if (ready) {
          clearInterval(interval);
        }
      } catch (err) {
        setInferenceReady(false);
      }
    }, 200);
    return () => clearInterval(interval);
  }, [hasSource, activeConfigId]);

  const feedUrl =
    activeConfigId != null
      ? `${API_URL}/video_feed?config=${encodeURIComponent(activeConfigId)}&_=${feedNonce}`
      : `${API_URL}/video_feed`;
  return (
    <div className="video-feed-container">
      <div
        className={`video-feed-frame ${hasSource ? '' : 'video-feed-frame--placeholder'}`}
      >
        {hasSource ? (
          <>
            <LiveFeedLabel />
            {/* No key: remounting created a NEW stream without reliably closing the old one,
                so rapid thumbnail switches stacked many /video_feed connections (see logs conn_id=*). */}
            <img
              ref={imgRef}
              src={feedUrl}
              alt="Video Feed"
              className={inferenceReady ? '' : 'video-feed-image-loading'}
            />
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
    </div>
  );
};

export default VideoPlayer;
