// assets/clientside.js
window.dash_clientside = Object.assign({}, window.dash_clientside, {
  ui: {
    // Return {compact: true/false} based on viewport width
    detectCompact: function (n) {
      const w = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0);
      // tweak 768 to your preferred md breakpoint
      return { compact: w < 768 };
    }
  }
});