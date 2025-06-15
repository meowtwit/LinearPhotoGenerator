let fileInput    = document.getElementById('fileInput');
let rhoInput     = document.getElementById('rhoInput');
let magVInput    = document.getElementById('magVInput');
let magWInput    = document.getElementById('magWInput');
let thetaInput   = document.getElementById('thetaInput');
let fillBtn      = document.getElementById('fillBtn');
let runBtn       = document.getElementById('runBtn');
let canvas       = document.getElementById('canvasOutput');
let ctx          = canvas.getContext('2d');
let downloadLink = document.getElementById('downloadLink');//documentプロパティはどこから？

// OpenCV.js 初期化完了
function onOpenCvReady() {
  fillBtn.disabled = false;
  runBtn.disabled  = false;
}

// 行列掛け算 (3x3 flat × flat)
function mult3(a, b) {
  let c = new Array(9);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let sum = 0;
      for (let k = 0; k < 3; k++) {
        sum += a[3*i + k] * b[3*k + j];
      }
      c[3*i + j] = sum;
    }
  }
  return c;
}

// 補完ボタン
fillBtn.addEventListener('click', () => {
  let p     = parseFloat(rhoInput.value);
  let vVal  = magVInput.value.trim();
  let wVal  = magWInput.value.trim();
  let tVal  = thetaInput.value.trim();

  if (isNaN(p)) {
    alert('ρ (内積の値) は必須入力です！');
    return;
  }

  // 入力済みフィールド数カウント
  let cntV = vVal  !== '' && !isNaN(parseFloat(vVal));
  let cntW = wVal  !== '' && !isNaN(parseFloat(wVal));
  let cntT = tVal  !== '' && !isNaN(parseFloat(tVal));
  let cnt  = [cntV, cntW, cntT].filter(x=>x).length;

  if (cnt !== 2) {
    alert('|v|, |w|, θ のうち「ちょうど２つ」を入力してください');
    return;
  }

  // 補完ロジック
  if (cntV && cntW) {
    if (p < (vVal*wVal)){
        // θ を補完
    let magV = parseFloat(vVal), magW = parseFloat(wVal);
    let cosT = p / (magV * magW);
    cosT = Math.max(-1, Math.min(1, cosT));
    let theta = Math.acos(cosT) * 180/Math.PI;
    thetaInput.value = theta.toFixed(2);
  }
  else{
    alert("|v|×|w|をp以上にしてください");
  }

    }
    
  else if (cntV && cntT) {
    // |w| を補完
    let magV = parseFloat(vVal), theta = parseFloat(tVal) * Math.PI/180;
    let denom = magV * Math.cos(theta);
    if (Math.abs(denom) < 1e-6) {
      alert('cosθ が 0 に近く、|w| を計算できません');
      return;
    }
    let magW = p / denom;
    magWInput.value = magW.toFixed(3);
  }
  else if (cntW && cntT) {
    // |v| を補完
    let magW = parseFloat(wVal), theta = parseFloat(tVal) * Math.PI/180;
    let denom = magW * Math.cos(theta);
    if (Math.abs(denom) < 1e-6) {
      alert('cosθ が 0 に近く、|v| を計算できません');
      return;
    }
    let magV = p / denom;
    magVInput.value = magV.toFixed(3);
  }
});

// 変換ボタン
runBtn.addEventListener('click', () => {
  if (!fileInput.files.length) {
    alert('画像を選択してください');
    return;
  }
  let p     = parseFloat(rhoInput.value);
  let magV  = parseFloat(magVInput.value);
  let magW  = parseFloat(magWInput.value);
  let theta = parseFloat(thetaInput.value) * Math.PI/180;

  if (isNaN(p) || isNaN(magV) || isNaN(magW) || isNaN(theta)) {
    alert('ρ, |v|, |w|, θ をすべて正しく入力してください');
    return;
  }

  // ファイル読み込み
  let reader = new FileReader();
  reader.onload = e => {
    let img = new Image();
    img.onload = () => {
      let src = cv.imread(img);
      let h0  = src.rows, w0 = src.cols;

      // ベクトル
      let v0  = magV, v1 = 0;
      let w0x = magW * Math.cos(theta), w1 = magW * Math.sin(theta);

      // 変換行列 H
      let Harr = [ v0, w0x, 0,  v1, w1, 0,  0,0,1 ];

      // 四隅変換で出力サイズ算出
      let xs = [], ys = [];
      [[0,0],[w0,0],[w0,h0],[0,h0]].forEach(([x,y]) => {
        xs.push(Harr[0]*x + Harr[1]*y + Harr[2]);
        ys.push(Harr[3]*x + Harr[4]*y + Harr[5]);
      });
      let minX = Math.min(...xs), maxX = Math.max(...xs);
      let minY = Math.min(...ys), maxY = Math.max(...ys);
      let outW = Math.ceil(maxX - minX);
      let outH = Math.ceil(maxY - minY);

      // 平行移動行列 T
      let Tarr = [1,0,-minX,  0,1,-minY,  0,0,1];
      let Htot = cv.matFromArray(3,3,cv.CV_64F, mult3(Tarr, Harr));

      // Warp
      let dst = new cv.Mat();
      cv.warpPerspective(src, dst, Htot, new cv.Size(outW,outH),
                         cv.INTER_LINEAR, cv.BORDER_CONSTANT,
                         new cv.Scalar(0,0,0,255));

      // 描画
      canvas.width  = outW;
      canvas.height = outH;
      cv.imshow(canvas, dst);

      // 矢印オーバーレイ
      ctx.lineWidth = 4;
      ctx.font      = "16px sans-serif";
      let ox = -minX, oy = -minY, scale = 100;

      // v (赤)
      ctx.strokeStyle = 'red';
      ctx.beginPath();
      ctx.moveTo(ox,oy);
      ctx.lineTo(v0*scale + ox, v1*scale + oy);
      ctx.stroke();
      ctx.fillStyle = 'red';
      ctx.fillText('v', v0*scale + ox + 5, v1*scale + oy + 5);

      // w (青)
      ctx.strokeStyle = 'blue';
      ctx.beginPath();
      ctx.moveTo(ox,oy);
      ctx.lineTo(w0x*scale + ox, w1*scale + oy);
      ctx.stroke();
      ctx.fillStyle = 'blue';
      ctx.fillText('w', w0x*scale + ox + 5, w1*scale + oy + 5);

      // cleanup
      src.delete(); dst.delete(); Htot.delete();

      // ダウンロードリンク
      let actualRho = v0*w0x + v1*w1;
      downloadLink.href = canvas.toDataURL('image/png');
      downloadLink.style.display = 'inline';
      downloadLink.textContent = `ダウンロード (実内積=${actualRho.toFixed(4)})`;
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(fileInput.files[0]);
});
