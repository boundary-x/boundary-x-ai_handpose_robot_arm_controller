import {
  HandLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

// --- [설정] 하드웨어 및 통신 설정 ---

const UUID_SERVICE = "6e400001-b5a3-f393-e0a9-e50e24dcca9e";
const UUID_RX = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"; 

// 서보모터 각도 제한
const LIMITS = {
    base: { min: 0, max: 180 },
    shoulder: { min: 20, max: 160 }, 
    elbow: { min: 20, max: 160 }     
};

// 스무딩 계수 (낮을수록 부드럽게 복귀)
const SMOOTHING = 0.15; 

// --- [변수] ---
let handLandmarker = undefined;
let webcam = null;
let canvas, ctx;
let lastVideoTime = -1;
let results = undefined;

let bluetoothDevice, rxCharacteristic;
let isConnected = false;
let isSendingData = false;

// 초기 목표/현재 각도 (원점)
let targetAngles = { b: 90, s: 90, e: 90, g: 0 };
let currentAngles = { b: 90, s: 90, e: 90 }; 

// DOM 요소
const modelStatus = document.getElementById("model-status");
const statusBt = document.getElementById("bt-status");
const packetLog = document.getElementById("packet-log");
const connectBtn = document.getElementById("connect-btn");
const disconnectBtn = document.getElementById("disconnect-btn");

const uiBars = {
    b: document.getElementById("bar-base"),
    s: document.getElementById("bar-shoulder"),
    e: document.getElementById("bar-elbow"),
};
const uiVals = {
    b: document.getElementById("val-base"),
    s: document.getElementById("val-shoulder"),
    e: document.getElementById("val-elbow"),
    g: document.getElementById("val-gripper"),
};
const trims = {
    b: document.getElementById("trim-base"),
    s: document.getElementById("trim-shoulder"),
};

// --- [1] AI 초기화 ---
async function createHandLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: 1 
  });
  modelStatus.innerText = "AI 모델 준비 완료";
  modelStatus.classList.add("ready");
  startWebcam();
}

// --- [2] 웹캠 ---
function startWebcam() {
  webcam = document.getElementById("webcam");
  canvas = document.getElementById("output_canvas");
  ctx = canvas.getContext("2d");
  const constraints = { video: { width: 1280, height: 720 } };
  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    webcam.srcObject = stream;
    webcam.addEventListener("loadeddata", predictWebcam);
  });
}

// --- [3] 메인 루프 (원점 복귀 로직 추가됨) ---
async function predictWebcam() {
  if (canvas.width !== webcam.videoWidth) {
    canvas.width = webcam.videoWidth;
    canvas.height = webcam.videoHeight;
  }

  let startTimeMs = performance.now();
  if (lastVideoTime !== webcam.currentTime) {
    lastVideoTime = webcam.currentTime;
    results = handLandmarker.detectForVideo(webcam, startTimeMs);
  }

  // 화면 그리기
  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(webcam, 0, 0, canvas.width, canvas.height);
  ctx.restore();

  // ★ [수정됨] 데이터 처리 및 원점 복귀 로직
  if (results.landmarks && results.landmarks.length > 0) {
    // 손이 인식되었을 때: 각도 계산 및 뼈대 그리기
    const landmarks = results.landmarks[0];
    calculateRobotAngles(landmarks); 
    drawSkeleton(landmarks);         
  } else {
    // 손이 인식되지 않았을 때: 원점(90, 90, 90, 0)으로 복귀 설정
    targetAngles.b = 90;
    targetAngles.s = 90;
    targetAngles.e = 90;
    targetAngles.g = 0;
  }

  smoothMove(); // 부드럽게 이동 (원점 복귀 시에도 적용됨)
  updateUI();   
  sendPacket(); // 전송

  window.requestAnimationFrame(predictWebcam);
}

// --- [4] 각도 계산 ---
function calculateRobotAngles(lm) {
    // 1. Base (좌우)
    let x = 1 - lm[0].x; 
    let baseRaw = map(x, 0, 1, LIMITS.base.max, LIMITS.base.min);
    
    // 2. Shoulder (거리)
    let size = getDistance(lm[0], lm[9]);
    let shoulderRaw = map(size, 0.05, 0.25, LIMITS.shoulder.min, LIMITS.shoulder.max);

    // 3. Elbow (상하/높이)
    let y = lm[0].y;
    let elbowRaw = map(y, 0, 1, LIMITS.elbow.max, LIMITS.elbow.min);

    // 4. Gripper
    let pinchDist = getDistance(lm[4], lm[8]);
    let gripState = (pinchDist < 0.05) ? 0 : 1; 

    // 5. Trim
    let trimB = parseInt(trims.b.value) || 0;
    let trimS = parseInt(trims.s.value) || 0;

    targetAngles.b = constrain(baseRaw + trimB, 0, 180);
    targetAngles.s = constrain(shoulderRaw + trimS, 0, 180);
    targetAngles.e = constrain(elbowRaw, 0, 180);
    targetAngles.g = gripState;
}

// --- [5] 유틸리티 ---
function smoothMove() {
    currentAngles.b += (targetAngles.b - currentAngles.b) * SMOOTHING;
    currentAngles.s += (targetAngles.s - currentAngles.s) * SMOOTHING;
    currentAngles.e += (targetAngles.e - currentAngles.e) * SMOOTHING;
}

function getDistance(p1, p2) {
    return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
}

function map(value, inMin, inMax, outMin, outMax) {
    return (value - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
}

function constrain(val, min, max) {
    return Math.min(Math.max(val, min), max);
}

function drawSkeleton(lm) {
    const w = canvas.width;
    const h = canvas.height;
    ctx.fillStyle = "#00E676"; 
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    [0, 4, 8, 9].forEach(i => {
        let cx = (1 - lm[i].x) * w;
        let cy = lm[i].y * h;
        ctx.beginPath(); ctx.arc(cx, cy, 6, 0, 2*Math.PI); ctx.fill(); ctx.stroke();
    });
    let tX = (1 - lm[4].x) * w; let tY = lm[4].y * h;
    let iX = (1 - lm[8].x) * w; let iY = lm[8].y * h;
    ctx.beginPath(); ctx.moveTo(tX, tY); ctx.lineTo(iX, iY);
    ctx.strokeStyle = targetAngles.g === 0 ? "#FF1744" : "#00E676";
    ctx.lineWidth = 4; ctx.stroke();
}

function updateUI() {
    let b = Math.round(currentAngles.b);
    let s = Math.round(currentAngles.s);
    let e = Math.round(currentAngles.e);
    uiVals.b.innerText = `${b}°`; uiVals.s.innerText = `${s}°`; uiVals.e.innerText = `${e}°`;
    uiBars.b.style.width = `${(b/180)*100}%`; uiBars.s.style.width = `${(s/180)*100}%`; uiBars.e.style.width = `${(e/180)*100}%`;
    if (targetAngles.g === 0) {
        uiVals.g.innerText = "CLOSE"; uiVals.g.style.color = "#FF1744";
    } else {
        uiVals.g.innerText = "OPEN"; uiVals.g.style.color = "#00E676";
    }
}

// --- [6] 통신 ---
async function sendPacket() {
    if (!isConnected || !rxCharacteristic || isSendingData) return;

    let b = String(Math.round(currentAngles.b)).padStart(3, '0');
    let s = String(Math.round(currentAngles.s)).padStart(3, '0');
    let e = String(Math.round(currentAngles.e)).padStart(3, '0');
    let g = String(targetAngles.g).padStart(3, '0');

    let packet = `B${b}S${s}E${e}G${g}`;
    packetLog.innerText = packet;

    try {
        isSendingData = true;
        const encoder = new TextEncoder();
        await rxCharacteristic.writeValue(encoder.encode(packet + "\r\n"));
    } catch (err) {
        // 전송 실패는 무시
    } finally {
        isSendingData = false;
    }
}

// --- 이벤트 리스너 ---
connectBtn.addEventListener('click', async () => {
  try {
    bluetoothDevice = await navigator.bluetooth.requestDevice({
      filters: [{ namePrefix: "BBC micro:bit" }],
      optionalServices: [UUID_SERVICE]
    });
    bluetoothDevice.addEventListener('gattserverdisconnected', onDisconnected);
    const server = await bluetoothDevice.gatt.connect();
    const service = await server.getPrimaryService(UUID_SERVICE);
    rxCharacteristic = await service.getCharacteristic(UUID_RX);

    isConnected = true;
    statusBt.innerText = "연결됨: " + bluetoothDevice.name;
    statusBt.classList.add("status-connected");
    connectBtn.classList.add("hidden");
    disconnectBtn.classList.remove("hidden");
  } catch (error) {
    alert("연결 실패: " + error);
  }
});

function onDisconnected() {
    isConnected = false;
    statusBt.innerText = "연결 해제됨";
    statusBt.classList.remove("status-connected");
    connectBtn.classList.remove("hidden");
    disconnectBtn.classList.add("hidden");
}

disconnectBtn.addEventListener('click', () => {
    if(bluetoothDevice && bluetoothDevice.gatt.connected) {
        bluetoothDevice.gatt.disconnect();
    }
});

createHandLandmarker();
