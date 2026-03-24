import React from 'react';
import './LogoBar.css';

const LogoBar = ({ onConfigClick }) => {
  return (
    <div className="logo-bar">
      <button
        className="logo-bar-config-btn"
        onClick={onConfigClick}
        type="button"
      >
        Config
      </button>
      <img src="/redhat-logo.png" alt="Red Hat Logo" className="logo redhat-logo" />
    </div>
  );
};

export default LogoBar;
