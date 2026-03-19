// script.js

// ===== Startup Splash Screen Loader =====
document.addEventListener("DOMContentLoaded", () => {
  document.body.classList.add("loading");
  
  window.addEventListener("load", () => {
    // Hide the loader after a short delay for a premium feel
    setTimeout(() => {
      document.body.classList.remove("loading");
      document.body.classList.add("loaded");
      
      const loader = document.getElementById("splash-loader");
      if(loader) {
          loader.classList.add("hidden");
      }
    }, 1500); // 1.5 seconds splash screen
  });
});