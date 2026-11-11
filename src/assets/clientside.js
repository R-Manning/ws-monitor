// assets/clientside.js
window.dash_clientside = Object.assign({}, window.dash_clientside, {
  ui: {
    detectCompact: function (n) {
      var w = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0);
      return { compact: w < 768 };
    }
  }
});

// ── WebRTC video stream (self-initializing, no Dash callback dependency) ──
(function () {
  var pc = null;
  var retryTimer = null;
  var retryDelayMs = 3000;
  var retryCount = 0;

  function connectWebRTC(video) {
    if (pc) {
      try { pc.close(); } catch (e) {}
      pc = null;
    }

    pc = new RTCPeerConnection({ iceServers: [] });

    pc.ontrack = function (event) {
      if (event.streams && event.streams[0]) {
        video.srcObject = event.streams[0];
        video.play().catch(function () {});
      }
    };

    pc.onconnectionstatechange = function () {
      var state = pc ? pc.connectionState : "closed";
      if (state === "connected") {
        retryCount = 0;
      } else if (state === "failed" || state === "disconnected" || state === "closed") {
        scheduleReconnect(video);
      }
    };

    fetch("/webrtc/offer")
      .then(function (res) {
        if (!res.ok) throw new Error("HTTP " + res.status);
        return res.json();
      })
      .then(function (offer) {
        if (offer.error) throw new Error(offer.error);
        return pc.setRemoteDescription(offer).then(function () { return offer; });
      })
      .then(function (offer) {
        return pc.createAnswer().then(function (answer) {
          return pc.setLocalDescription(answer).then(function () {
            return { sdp: pc.localDescription.sdp, type: pc.localDescription.type, pc_id: offer.pc_id };
          });
        });
      })
      .then(function (answer) {
        return fetch("/webrtc/answer", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(answer)
        });
      })
      .then(function (res) {
        if (!res.ok) throw new Error("HTTP " + res.status);
        return res.json();
      })
      .then(function (body) {
        if (body.error) throw new Error(body.error);
      })
      .catch(function (err) {
        scheduleReconnect(video);
      });
  }

  function scheduleReconnect(video) {
    if (pc) {
      try { pc.close(); } catch (e) {}
      pc = null;
    }
    retryCount += 1;
    var delay = Math.min(retryDelayMs * retryCount, 15000);
    clearTimeout(retryTimer);
    retryTimer = setTimeout(function () { connectWebRTC(video); }, delay);
  }

  function tryConnect() {
    var video = document.getElementById("video");
    if (!video) {
      setTimeout(tryConnect, 200);
      return;
    }
    video.setAttribute("playsinline", "");
    connectWebRTC(video);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () { setTimeout(tryConnect, 500); });
  } else {
    setTimeout(tryConnect, 500);
  }
})();