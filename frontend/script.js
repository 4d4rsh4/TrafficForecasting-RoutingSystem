document.addEventListener("DOMContentLoaded", () => {
  document.body.classList.add("loading");

  window.addEventListener("load", () => {
    setTimeout(() => {
      document.body.classList.remove("loading");
      document.body.classList.add("loaded");

      const loader = document.getElementById("splash-loader");
      if (loader) {
        loader.classList.add("hidden");
      }
    }, 1500);
  });
});
