"""Generates a standalone interactive Three.js 3D Camera Trajectory Visualizer HTML.
Embeds exported 300-frame trajectory data directly into the HTML file.
"""

import json
import os


def generate_html():
    json_path = "/root/.gemini/antigravity-cli/scratch/camera_engine/trajectories_export.json"
    html_out = "/root/.gemini/antigravity-cli/scratch/camera_engine/camera_visualizer.html"
    
    with open(json_path, "r") as f:
        data = json.load(f)
        
    json_str = json.dumps(data)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Stage 1: Camera Trajectory Engine - Interactive 3D Visualizer</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
  <style>
    * {{
      margin: 0;
      padding: 0;
      box-sizing: border-box;
      user-select: none;
    }}
    body {{
      background: #08090d;
      color: #e2e8f0;
      font-family: 'Inter', -apple-system, sans-serif;
      overflow: hidden;
      width: 100vw;
      height: 100vh;
    }}
    #canvas-container {{
      width: 100%;
      height: 100%;
      position: absolute;
      top: 0;
      left: 0;
    }}
    /* Top Bar / Header */
    .top-bar {{
      position: absolute;
      top: 20px;
      left: 20px;
      right: 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      pointer-events: none;
      z-index: 10;
    }}
    .badge-card {{
      background: rgba(15, 23, 42, 0.75);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      padding: 12px 20px;
      border-radius: 12px;
      pointer-events: auto;
      box-shadow: 0 10px 25px rgba(0,0,0,0.4);
    }}
    .badge-title {{
      font-size: 15px;
      font-weight: 800;
      letter-spacing: -0.02em;
      color: #38bdf8;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .badge-sub {{
      font-size: 11px;
      color: #94a3b8;
      font-family: 'JetBrains Mono', monospace;
      margin-top: 2px;
    }}
    /* View Switcher Controls */
    .view-toggle-group {{
      display: flex;
      gap: 8px;
      pointer-events: auto;
    }}
    .btn-toggle {{
      background: rgba(30, 41, 59, 0.8);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255, 255, 255, 0.12);
      color: #cbd5e1;
      padding: 10px 18px;
      border-radius: 10px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .btn-toggle:hover {{
      background: rgba(51, 65, 85, 0.9);
      color: #ffffff;
      border-color: rgba(56, 189, 248, 0.4);
    }}
    .btn-toggle.active {{
      background: linear-gradient(135deg, #0284c7, #0369a1);
      color: #ffffff;
      border-color: #38bdf8;
      box-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
    }}
    /* Telemetry HUD Panel (Left side) */
    .hud-panel {{
      position: absolute;
      top: 90px;
      left: 20px;
      width: 320px;
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(14px);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 14px;
      padding: 16px;
      z-index: 10;
      box-shadow: 0 15px 30px rgba(0,0,0,0.5);
    }}
    .hud-header {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #64748b;
      font-weight: 700;
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
    }}
    .shot-select-dropdown {{
      width: 100%;
      background: #0f172a;
      border: 1px solid #334155;
      color: #38bdf8;
      font-size: 13px;
      font-weight: 600;
      padding: 9px 12px;
      border-radius: 8px;
      margin-bottom: 14px;
      outline: none;
      cursor: pointer;
    }}
    .prompt-box {{
      background: rgba(2, 6, 23, 0.6);
      border-left: 3px solid #38bdf8;
      padding: 8px 10px;
      font-size: 11px;
      line-height: 1.45;
      color: #cbd5e1;
      border-radius: 0 6px 6px 0;
      margin-bottom: 14px;
      font-style: italic;
    }}
    .telemetry-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
    }}
    .telemetry-label {{
      color: #94a3b8;
    }}
    .telemetry-val {{
      color: #f8fafc;
      font-weight: 600;
    }}
    .telemetry-val.highlight {{
      color: #38bdf8;
    }}
    .telemetry-val.accent {{
      color: #ec4899;
    }}
    /* Bottom Controls */
    .bottom-bar {{
      position: absolute;
      bottom: 24px;
      left: 20px;
      right: 20px;
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(14px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 16px;
      padding: 16px 24px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      z-index: 10;
      box-shadow: 0 15px 35px rgba(0,0,0,0.6);
    }}
    .slider-track {{
      display: flex;
      align-items: center;
      gap: 16px;
    }}
    .timeline-slider {{
      flex: 1;
      height: 6px;
      -webkit-appearance: none;
      background: #334155;
      border-radius: 3px;
      outline: none;
      cursor: pointer;
    }}
    .timeline-slider::-webkit-slider-thumb {{
      -webkit-appearance: none;
      width: 18px;
      height: 18px;
      border-radius: 50%;
      background: #38bdf8;
      cursor: pointer;
      box-shadow: 0 0 10px #38bdf8;
      transition: transform 0.1s;
    }}
    .timeline-slider::-webkit-slider-thumb:hover {{
      transform: scale(1.2);
    }}
    .playback-controls {{
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .ctrl-buttons {{
      display: flex;
      gap: 10px;
      align-items: center;
    }}
    .btn-ctrl {{
      background: #1e293b;
      border: 1px solid #475569;
      color: #f8fafc;
      padding: 8px 16px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
    }}
    .btn-ctrl:hover {{
      background: #334155;
    }}
    .speed-pills {{
      display: flex;
      gap: 6px;
      background: #0f172a;
      padding: 4px;
      border-radius: 8px;
      border: 1px solid #334155;
    }}
    .btn-pill {{
      background: transparent;
      border: none;
      color: #94a3b8;
      font-size: 11px;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 6px;
      cursor: pointer;
    }}
    .btn-pill.active {{
      background: #38bdf8;
      color: #0f172a;
    }}
    .pov-crosshair {{
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      width: 20px;
      height: 20px;
      pointer-events: none;
      display: none;
      z-index: 5;
    }}
    .pov-crosshair::before, .pov-crosshair::after {{
      content: '';
      position: absolute;
      background: rgba(255, 255, 255, 0.4);
    }}
    .pov-crosshair::before {{
      top: 9px;
      left: 0;
      width: 20px;
      height: 2px;
    }}
    .pov-crosshair::after {{
      top: 0;
      left: 9px;
      width: 2px;
      height: 20px;
    }}
  </style>
</head>
<body>

  <div id="canvas-container"></div>
  <div class="pov-crosshair" id="pov-crosshair"></div>

  <!-- Top Bar -->
  <div class="top-bar">
    <div class="badge-card">
      <div class="badge-title">
        <span style="font-size: 18px;">🎥</span> Stage 1: Camera Trajectory Engine (27.21M)
      </div>
      <div class="badge-sub">Lie-Algebra Cumulative B-Spline SE(3) Visualizer • 8K 300 Frames</div>
    </div>

    <div class="view-toggle-group">
      <button class="btn-toggle active" id="btn-god-view" onclick="setViewMode('god')">
        🌐 Director 3rd Person View
      </button>
      <button class="btn-toggle" id="btn-pov-view" onclick="setViewMode('pov')">
        🎬 Camera Lens POV (Optical View)
      </button>
    </div>
  </div>

  <!-- HUD Telemetry Panel -->
  <div class="hud-panel">
    <div class="hud-header">
      <span>Directorial Shot</span>
      <span style="color: #10b981;">● Online</span>
    </div>
    
    <select class="shot-select-dropdown" id="shot-select" onchange="switchShot(this.value)">
      <option value="orbit">1. 360° Cinematic Orbit (50mm)</option>
      <option value="vertigo">2. Hitchcock Vertigo Dolly-Zoom</option>
      <option value="spiral">3. Ascending Inward Spiral (35mm)</option>
      <option value="fpv">4. FPV Aerial Reveal Shot</option>
    </select>

    <div class="prompt-box" id="prompt-box">
      "Cinematic clockwise 360 degree orbit around frozen center with 50mm lens"
    </div>

    <div class="hud-header">Real-Time Optics & Kinematics</div>

    <div class="telemetry-row">
      <span class="telemetry-label">Timeline Frame:</span>
      <span class="telemetry-val highlight" id="hud-frame">1 / 300 (0.00s)</span>
    </div>
    <div class="telemetry-row">
      <span class="telemetry-label">Lens Focal (mm):</span>
      <span class="telemetry-val highlight" id="hud-focal">50.0 mm</span>
    </div>
    <div class="telemetry-row">
      <span class="telemetry-label">Horizontal FOV:</span>
      <span class="telemetry-val" id="hud-fov">39.6°</span>
    </div>
    <div class="telemetry-row">
      <span class="telemetry-label">Camera Position:</span>
      <span class="telemetry-val" id="hud-pos">[+0.35, -1.39, +1.18]</span>
    </div>
    <div class="telemetry-row">
      <span class="telemetry-label">Distance to Hero:</span>
      <span class="telemetry-val accent" id="hud-dist">1.43 m</span>
    </div>
    <div class="telemetry-row">
      <span class="telemetry-label">Resolution Scale:</span>
      <span class="telemetry-val">8K (7680 x 4320)</span>
    </div>
  </div>

  <!-- Bottom Playback Control Bar -->
  <div class="bottom-bar">
    <div class="slider-track">
      <span style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #94a3b8;" id="time-display">0.00s</span>
      <input type="range" min="0" max="299" value="0" class="timeline-slider" id="timeline" oninput="onSliderInput(this.value)">
      <span style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #94a3b8;">10.00s</span>
    </div>

    <div class="playback-controls">
      <div class="ctrl-buttons">
        <button class="btn-ctrl" id="btn-play" onclick="togglePlay()">⏸ Pause</button>
        <button class="btn-ctrl" onclick="resetPlayback()">⏮ Restart</button>
      </div>

      <div class="speed-pills">
        <button class="btn-pill" onclick="setSpeed(0.5, this)">0.5x</button>
        <button class="btn-pill active" onclick="setSpeed(1.0, this)">1.0x</button>
        <button class="btn-pill" onclick="setSpeed(2.0, this)">2.0x</button>
      </div>
    </div>
  </div>

  <script>
    // Injected Trajectory Data from Trained PyTorch Model
    const trajectoryData = {json_str};

    let currentShotId = 'orbit';
    let currentFrame = 0;
    let isPlaying = true;
    let playbackSpeed = 1.0;
    let viewMode = 'god'; // 'god' or 'pov'

    // Three.js Core Variables
    let scene, renderer, godCamera, povCamera, controls;
    let trajectoryLine, cameraFrustumHelper, keyframeSpheres = [];
    let heroGroup;

    function init() {{
      const container = document.getElementById('canvas-container');
      const w = window.innerWidth;
      const h = window.innerHeight;

      // 1. Scene & Renderer
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x07090e);
      scene.fog = new THREE.FogExp2(0x07090e, 0.08);

      renderer = new THREE.WebGLRenderer({{ antialias: true }});
      renderer.setPixelRatio(window.devicePixelRatio);
      renderer.setSize(w, h);
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      container.appendChild(renderer.domElement);

      // 2. Cameras
      godCamera = new THREE.PerspectiveCamera(50, w / h, 0.1, 100);
      godCamera.position.set(2.8, -3.2, 2.5);

      povCamera = new THREE.PerspectiveCamera(50, w / h, 0.05, 100);

      // 3. OrbitControls for Director View
      controls = new THREE.OrbitControls(godCamera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controls.target.set(0, 0, 1.0);

      // 4. Studio Lighting
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
      scene.add(ambientLight);

      const keyLight = new THREE.DirectionalLight(0x38bdf8, 1.2);
      keyLight.position.set(4, -3, 5);
      keyLight.castShadow = true;
      scene.add(keyLight);

      const rimLight = new THREE.DirectionalLight(0xec4899, 1.5);
      rimLight.position.set(-4, 4, 3);
      scene.add(rimLight);

      // 5. Ground Grid
      const grid = new THREE.GridHelper(10, 20, 0x0284c7, 0x1e293b);
      grid.rotation.x = Math.PI / 2;
      scene.add(grid);

      // 6. Build Central Frozen Hero Statue
      buildHeroStatue();

      // 7. Build Camera Frustum Rig
      buildCameraRig();

      // 8. Load Active Shot
      loadShot(currentShotId);

      window.addEventListener('resize', onWindowResize);
      animate();
    }}

    function buildHeroStatue() {{
      heroGroup = new THREE.Group();

      // Pedestal
      const baseGeo = new THREE.CylinderGeometry(0.5, 0.6, 0.3, 32);
      const baseMat = new THREE.MeshStandardMaterial({{ color: 0x1e293b, roughness: 0.8, metalness: 0.2 }});
      const baseMesh = new THREE.Mesh(baseGeo, baseMat);
      baseMesh.rotation.x = Math.PI / 2;
      baseMesh.position.set(0, 0, 0.15);
      heroGroup.add(baseMesh);

      // Central Figure Sculpture
      const torsoGeo = new THREE.ConeGeometry(0.35, 1.2, 16);
      const torsoMat = new THREE.MeshStandardMaterial({{
        color: 0x94a3b8,
        roughness: 0.3,
        metalness: 0.7,
        wireframe: false
      }});
      const torsoMesh = new THREE.Mesh(torsoGeo, torsoMat);
      torsoMesh.rotation.x = Math.PI / 2;
      torsoMesh.position.set(0, 0, 0.9);
      heroGroup.add(torsoMesh);

      // Hero Head with Horns (The geometric feature user highlighted!)
      const headGeo = new THREE.SphereGeometry(0.18, 24, 24);
      const headMat = new THREE.MeshStandardMaterial({{ color: 0x38bdf8, metalness: 0.8, roughness: 0.2 }});
      const headMesh = new THREE.Mesh(headGeo, headMat);
      headMesh.position.set(0, 0, 1.55);
      heroGroup.add(headMesh);

      // Horn Left
      const hornGeo = new THREE.ConeGeometry(0.06, 0.45, 12);
      const hornMat = new THREE.MeshStandardMaterial({{ color: 0xec4899, metalness: 0.9, roughness: 0.1 }});
      const hornL = new THREE.Mesh(hornGeo, hornMat);
      hornL.position.set(-0.18, 0, 1.75);
      hornL.rotation.z = 0.5;
      hornL.rotation.x = Math.PI / 2;
      heroGroup.add(hornL);

      // Horn Right
      const hornR = new THREE.Mesh(hornGeo, hornMat);
      hornR.position.set(0.18, 0, 1.75);
      hornR.rotation.z = -0.5;
      hornR.rotation.x = Math.PI / 2;
      heroGroup.add(hornR);

      scene.add(heroGroup);
    }}

    function buildCameraRig() {{
      cameraFrustumHelper = new THREE.Group();

      // Camera Body
      const bodyGeo = new THREE.BoxGeometry(0.12, 0.18, 0.08);
      const bodyMat = new THREE.MeshStandardMaterial({{ color: 0x0f172a, roughness: 0.5, metalness: 0.8 }});
      const body = new THREE.Mesh(bodyGeo, bodyMat);
      cameraFrustumHelper.add(body);

      // Lens Cylinder
      const lensGeo = new THREE.CylinderGeometry(0.045, 0.055, 0.08, 16);
      const lensMat = new THREE.MeshStandardMaterial({{ color: 0x38bdf8, metalness: 0.9, roughness: 0.1 }});
      const lens = new THREE.Mesh(lensGeo, lensMat);
      lens.position.set(0, 0, -0.06);
      lens.rotation.x = Math.PI / 2;
      cameraFrustumHelper.add(lens);

      // Frustum Wireframe Pyramid
      const frustumGeo = new THREE.ConeGeometry(0.2, 0.4, 4, 1, true);
      const frustumMat = new THREE.MeshBasicMaterial({{
        color: 0x38bdf8,
        wireframe: true,
        transparent: true,
        opacity: 0.6
      }});
      const frustum = new THREE.Mesh(frustumGeo, frustumMat);
      frustum.position.set(0, 0, -0.24);
      frustum.rotation.x = -Math.PI / 2;
      frustum.rotation.y = Math.PI / 4;
      cameraFrustumHelper.add(frustum);

      scene.add(cameraFrustumHelper);
    }}

    function loadShot(shotId) {{
      currentShotId = shotId;
      const shot = trajectoryData.shots[shotId];

      document.getElementById('prompt-box').innerText = `"${{shot.prompt}}"`;

      // 1. Remove old trajectory line and keyframe spheres
      if (trajectoryLine) scene.remove(trajectoryLine);
      keyframeSpheres.forEach(s => scene.remove(s));
      keyframeSpheres = [];

      // 2. Build 3D Trajectory Tube / Spline
      const points = shot.positions.map(p => new THREE.Vector3(p[0], p[1], p[2]));
      const curve = new THREE.CatmullRomCurve3(points);
      const tubeGeo = new THREE.TubeGeometry(curve, 300, 0.012, 8, false);
      const tubeMat = new THREE.MeshStandardMaterial({{
        color: 0x38bdf8,
        emissive: 0x0284c7,
        emissiveIntensity: 0.6,
        roughness: 0.3,
        metalness: 0.8
      }});
      trajectoryLine = new THREE.Mesh(tubeGeo, tubeMat);
      scene.add(trajectoryLine);

      // 3. Keyframe Milestone Orbs (every 50 frames)
      const orbGeo = new THREE.SphereGeometry(0.03, 16, 16);
      const orbMat = new THREE.MeshBasicMaterial({{ color: 0xec4899 }});
      for (let i = 0; i < 300; i += 50) {{
        const orb = new THREE.Mesh(orbGeo, orbMat);
        const p = shot.positions[i];
        orb.position.set(p[0], p[1], p[2]);
        scene.add(orb);
        keyframeSpheres.push(orb);
      }}

      currentFrame = 0;
      updateFrame(0);
    }}

    function switchShot(shotId) {{
      loadShot(shotId);
    }}

    function setViewMode(mode) {{
      viewMode = mode;
      document.getElementById('btn-god-view').classList.toggle('active', mode === 'god');
      document.getElementById('btn-pov-view').classList.toggle('active', mode === 'pov');
      document.getElementById('pov-crosshair').style.display = (mode === 'pov') ? 'block' : 'none';
      cameraFrustumHelper.visible = (mode === 'god');
    }}

    function updateFrame(frameIdx) {{
      const shot = trajectoryData.shots[currentShotId];
      if (!shot) return;

      const pos = shot.positions[frameIdx];
      const quat = shot.quaternions[frameIdx]; // [w, x, y, z]
      const focal = shot.focal_mm[frameIdx];
      const fov = shot.fov_deg[frameIdx];

      // Update Rig Position and Orientation
      cameraFrustumHelper.position.set(pos[0], pos[1], pos[2]);
      cameraFrustumHelper.quaternion.set(quat[1], quat[2], quat[3], quat[0]);

      // Update POV Camera
      povCamera.position.set(pos[0], pos[1], pos[2]);
      povCamera.quaternion.set(quat[1], quat[2], quat[3], quat[0]);
      povCamera.fov = fov;
      povCamera.updateProjectionMatrix();

      // Telemetry HUD updates
      const tSec = (frameIdx / 30.0).toFixed(2);
      document.getElementById('hud-frame').innerText = `${{frameIdx + 1}} / 300 (${{tSec}}s)`;
      document.getElementById('hud-focal').innerText = `${{focal.toFixed(1)}} mm`;
      document.getElementById('hud-fov').innerText = `${{fov.toFixed(1)}}°`;
      document.getElementById('hud-pos').innerText = `[${{pos[0].toFixed(2)}}, ${{pos[1].toFixed(2)}}, ${{pos[2].toFixed(2)}}]`;
      
      const distToHero = Math.sqrt(pos[0]*pos[0] + pos[1]*pos[1] + (pos[2]-1.0)*(pos[2]-1.0));
      document.getElementById('hud-dist').innerText = `${{distToHero.toFixed(2)}} m`;

      document.getElementById('time-display').innerText = `${{tSec}}s`;
      document.getElementById('timeline').value = frameIdx;
    }}

    function onSliderInput(val) {{
      currentFrame = parseInt(val);
      updateFrame(currentFrame);
    }}

    function togglePlay() {{
      isPlaying = !isPlaying;
      document.getElementById('btn-play').innerText = isPlaying ? '⏸ Pause' : '▶ Play';
    }}

    function resetPlayback() {{
      currentFrame = 0;
      updateFrame(0);
    }}

    function setSpeed(speed, btn) {{
      playbackSpeed = speed;
      document.querySelectorAll('.btn-pill').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    }}

    let lastTime = performance.now();
    let frameAccum = 0;

    function animate() {{
      requestAnimationFrame(animate);

      const now = performance.now();
      const dt = (now - lastTime) / 1000.0;
      lastTime = now;

      if (isPlaying) {{
        frameAccum += dt * 30.0 * playbackSpeed;
        if (frameAccum >= 1.0) {{
          const advance = Math.floor(frameAccum);
          frameAccum -= advance;
          currentFrame = (currentFrame + advance) % 300;
          updateFrame(currentFrame);
        }}
      }}

      controls.update();

      const activeCam = (viewMode === 'pov') ? povCamera : godCamera;
      renderer.render(scene, activeCam);
    }}

    function onWindowResize() {{
      const w = window.innerWidth;
      const h = window.innerHeight;
      renderer.setSize(w, h);
      godCamera.aspect = w / h;
      godCamera.updateProjectionMatrix();
      povCamera.aspect = w / h;
      povCamera.updateProjectionMatrix();
    }}

    window.onload = init;
  </script>
</body>
</html>
"""
    with open(html_out, "w") as f:
        f.write(html_content)
    print(f"[+] Successfully generated standalone visualizer: {html_out}")


if __name__ == "__main__":
    generate_html()
