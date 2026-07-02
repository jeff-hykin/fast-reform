// Plain-JS loader + canvas renderer for the fast-reform wasm module. No build
// step: fetch the .wasm, call the raw C-ABI exports, read flat f32 arrays out of
// linear memory, and draw with the 2D canvas context.

const wasm = await loadWasm("./fast_reform.wasm")

const SEED = 7
const SHAPE_CIRCLE = 0
const SHAPE_SQUARE = 1

const canvas = document.getElementById("canvas")
const context = canvas.getContext("2d")
const alphaSlider = document.getElementById("alpha")
const alphaLabel = document.getElementById("alphaLabel")
const nodesSlider = document.getElementById("nodes")
const nodesLabel = document.getElementById("nodesLabel")
const overlay = document.getElementById("overlay")
const playButton = document.getElementById("play")
const circleButton = document.getElementById("shapeCircle")
const squareButton = document.getElementById("shapeSquare")

// Scene-dependent state — rebuilt by initScene whenever the shape changes.
let pointCount = 0
let maxNodes = 0
let pointTimes = new Float32Array(0)
let worldBounds = { minX: -1, minY: -1, maxX: 1, maxY: 1 }

function initScene(shape) {
    pointCount = wasm.fr_init(SEED, shape)
    maxNodes = wasm.fr_max_nodes()
    // Per-point normalized time (constant across alpha) drives the rainbow color.
    pointTimes = readF32(wasm.fr_point_times_ptr(), pointCount)
    // Fix the world→screen transform from the union of the open-loop and
    // closed-loop extents so the view never jumps as the slider moves.
    worldBounds = unionBounds(extentsAt(0), extentsAt(1))

    nodesSlider.max = maxNodes
    nodesSlider.value = maxNodes
    nodesLabel.textContent = maxNodes
}

initScene(SHAPE_CIRCLE)

let devicePixelScale = 1
function resize() {
    devicePixelScale = window.devicePixelRatio || 1
    const stage = document.getElementById("stage")
    canvas.width = Math.floor(stage.clientWidth * devicePixelScale)
    canvas.height = Math.floor(stage.clientHeight * devicePixelScale)
    draw()
}
window.addEventListener("resize", resize)

function draw() {
    render(parseFloat(alphaSlider.value), parseInt(nodesSlider.value, 10))
}

function render(alpha, nodeTarget) {
    wasm.fr_warp(alpha, nodeTarget)
    const nodeCount = wasm.fr_num_nodes()
    const points = readF32(wasm.fr_points_ptr(), pointCount * 2)
    const nodes = readF32(wasm.fr_node_points_ptr(), nodeCount * 2)

    const project = makeProjection(canvas.width, canvas.height, worldBounds)

    context.fillStyle = "#0d1117"
    context.fillRect(0, 0, canvas.width, canvas.height)

    // Lidar points, colored rainbow by observation time.
    const radius = 1.6 * devicePixelScale
    for (let index = 0; index < pointCount; index++) {
        const [screenX, screenY] = project(points[index * 2], points[index * 2 + 1])
        context.fillStyle = rainbow(pointTimes[index])
        context.beginPath()
        context.arc(screenX, screenY, radius, 0, Math.PI * 2)
        context.fill()
    }

    // Pose-graph edges: consecutive nodes, plus the closing edge back to node 0.
    context.strokeStyle = "rgba(201, 209, 217, 0.55)"
    context.lineWidth = 1.4 * devicePixelScale
    context.beginPath()
    for (let index = 0; index < nodeCount; index++) {
        const [screenX, screenY] = project(nodes[index * 2], nodes[index * 2 + 1])
        if (index === 0) { context.moveTo(screenX, screenY) } else { context.lineTo(screenX, screenY) }
    }
    context.stroke()
    // Closing edge (the loop) — dashed to distinguish it.
    const [firstX, firstY] = project(nodes[0], nodes[1])
    const [lastX, lastY] = project(nodes[(nodeCount - 1) * 2], nodes[(nodeCount - 1) * 2 + 1])
    context.setLineDash([6 * devicePixelScale, 5 * devicePixelScale])
    context.beginPath()
    context.moveTo(lastX, lastY)
    context.lineTo(firstX, firstY)
    context.stroke()
    context.setLineDash([])

    // Pose-graph nodes.
    for (let index = 0; index < nodeCount; index++) {
        const [screenX, screenY] = project(nodes[index * 2], nodes[index * 2 + 1])
        context.fillStyle = "#c9d1d9"
        context.beginPath()
        context.arc(screenX, screenY, 3 * devicePixelScale, 0, Math.PI * 2)
        context.fill()
    }

    // Highlight the two seam ends so the overlap is obvious.
    markEnd(context, project(nodes[0], nodes[1]), "#3fb950", "start")
    markEnd(context, project(nodes[(nodeCount - 1) * 2], nodes[(nodeCount - 1) * 2 + 1]), "#f85149", "end")

    const gap = Math.hypot(firstX - lastX, firstY - lastY) / devicePixelScale
    overlay.innerHTML =
        `points: <code>${pointCount}</code> &nbsp; nodes: <code>${nodeCount}</code> / <code>${maxNodes}</code><br>` +
        `closure: <code>${Math.round(alpha * 100)}%</code> &nbsp; seam gap: <code>${gap.toFixed(0)}px</code>`
    alphaLabel.textContent = `${Math.round(alpha * 100)}%`
    nodesLabel.textContent = `${nodeCount}`
}

function markEnd(context, [screenX, screenY], color, label) {
    context.strokeStyle = color
    context.lineWidth = 2 * devicePixelScale
    context.beginPath()
    context.arc(screenX, screenY, 7 * devicePixelScale, 0, Math.PI * 2)
    context.stroke()
    context.fillStyle = color
    context.font = `${11 * devicePixelScale}px system-ui`
    context.fillText(label, screenX + 10 * devicePixelScale, screenY - 8 * devicePixelScale)
}

alphaSlider.addEventListener("input", () => {
    stopPlaying()
    draw()
})

nodesSlider.addEventListener("input", () => {
    draw()
})

function selectShape(shape, activeButton, inactiveButton) {
    stopPlaying()
    activeButton.classList.add("active")
    inactiveButton.classList.remove("active")
    alphaSlider.value = 0
    initScene(shape)
    draw()
}
circleButton.addEventListener("click", () => selectShape(SHAPE_CIRCLE, circleButton, squareButton))
squareButton.addEventListener("click", () => selectShape(SHAPE_SQUARE, squareButton, circleButton))

// ---- Autoplay: sweep the slider open → closed and back ----
let playTimer = null
let playDirection = 1
function stopPlaying() {
    if (playTimer !== null) {
        cancelAnimationFrame(playTimer)
        playTimer = null
        playButton.textContent = "▶ Play"
    }
}
function playStep() {
    let value = parseFloat(alphaSlider.value) + playDirection * 0.006
    if (value >= 1) { value = 1; playDirection = -1 }
    else if (value <= 0) { value = 0; playDirection = 1 }
    alphaSlider.value = value
    render(value, parseInt(nodesSlider.value, 10))
    playTimer = requestAnimationFrame(playStep)
}
playButton.addEventListener("click", () => {
    if (playTimer !== null) { stopPlaying() } else {
        playButton.textContent = "⏸ Pause"
        playTimer = requestAnimationFrame(playStep)
    }
})

resize()

// ---------------- helpers ----------------

async function loadWasm(url) {
    const response = await fetch(url)
    const { instance } = await WebAssembly.instantiate(await response.arrayBuffer(), {})
    return instance.exports
}

// Read `count` f32 values starting at byte offset `ptr`. Must be re-read after
// any wasm call that may have grown memory.
function readF32(ptr, count) {
    return new Float32Array(wasm.memory.buffer, ptr, count).slice()
}

function extentsAt(alpha) {
    wasm.fr_warp(alpha, maxNodes)
    const points = readF32(wasm.fr_points_ptr(), pointCount * 2)
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
    for (let index = 0; index < pointCount; index++) {
        const x = points[index * 2]
        const y = points[index * 2 + 1]
        if (x < minX) minX = x
        if (y < minY) minY = y
        if (x > maxX) maxX = x
        if (y > maxY) maxY = y
    }
    return { minX, minY, maxX, maxY }
}

function unionBounds(a, b) {
    return {
        minX: Math.min(a.minX, b.minX),
        minY: Math.min(a.minY, b.minY),
        maxX: Math.max(a.maxX, b.maxX),
        maxY: Math.max(a.maxY, b.maxY),
    }
}

function makeProjection(width, height, bounds) {
    const padding = 0.08
    const worldWidth = (bounds.maxX - bounds.minX) || 1
    const worldHeight = (bounds.maxY - bounds.minY) || 1
    const scale = Math.min(
        (width * (1 - 2 * padding)) / worldWidth,
        (height * (1 - 2 * padding)) / worldHeight,
    )
    const centerX = (bounds.minX + bounds.maxX) / 2
    const centerY = (bounds.minY + bounds.maxY) / 2
    return (worldX, worldY) => [
        width / 2 + (worldX - centerX) * scale,
        // Flip Y so +Y is up on screen.
        height / 2 - (worldY - centerY) * scale,
    ]
}

// Time in [0,1] → rainbow (red → violet).
function rainbow(t) {
    const hue = (1 - t) * 280
    return `hsl(${hue}, 85%, 60%)`
}
