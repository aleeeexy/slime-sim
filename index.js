import { GUI } from "https://threejsfundamentals.org/3rdparty/dat.gui.module.js";
const VertShader = `
    @vertex fn vs(@builtin(vertex_index) vertexIndex : u32) -> @builtin(position) vec4f {
        let pos = array(
            vec2f(-1.0, -1.0),
            vec2f(-1.0, 3.0),
            vec2f(3.0, -1.0)
        );
        return vec4f(pos[vertexIndex], 0.0, 1.0);
    }
`;

const FragShader = `
    @group(0) @binding(0) var trailTex: texture_2d<f32>;
    @group(0) @binding(1) var texSampler: sampler;
    @group(0) @binding(2) var<uniform> params: vec4f;
    @group(0) @binding(3) var<uniform> colors: vec3f;
    @group(0) @binding(4) var foodTex: texture_2d<f32>;

    @fragment fn fs(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
        let uv = fragCoord.xy / vec2f(params.xy);

        let trail = textureSample(trailTex, texSampler, uv).x;
        let foodColor = textureSample(foodTex, texSampler, uv); // This is a vec4f

        // Trail as grayscale colored by user settings
        let trailColor = vec3f(
            trail * colors.x / 255.0,
            trail * colors.y / 255.0,
            trail * colors.z / 255.0
        );

        // Blend: trail + food (clamped)
        let finalColor = clamp(trailColor + foodColor.rgb, vec3f(0.0), vec3f(1.0));

        return vec4f(finalColor, 1.0);
    }
`;

const AgentShader = `
    struct Agent {
        pos: vec2f,
        angle: f32,
    }

    @group(0) @binding(0) var<storage, read_write> agents: array<Agent>;
    @group(0) @binding(1) var trailTex: texture_2d<f32>;
    @group(0) @binding(2) var trailOut: texture_storage_2d<rgba8unorm, write>;
    @group(0) @binding(3) var<uniform> params: vec4f;
    @group(0) @binding(4) var<uniform> simParams: vec4f;
	  @group(0) @binding(5) var foodTex: texture_2d<f32>;
    
    fn random(seed: f32) -> f32 {
        return fract(sin(seed * 78.233) * 43758.5453);
    }
    
    fn wrap(value: f32, max: f32) -> f32 {
        return clamp(value, 0.0, max - 1.0);
    }
    
    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) id: vec3u) {
        let i = id.x;
        if (i >= arrayLength(&agents)) { return; }
        
        var agent = agents[i];
        let width = params.x;
        let height = params.y;
        let sensorAngle = simParams.x;
        let sensorDist = simParams.y;
        let moveSpeed = simParams.z;
        let turnSpeed = simParams.w;
        
        // Sensor positions
        let front = agent.pos + vec2f(cos(agent.angle), sin(agent.angle)) * sensorDist;
        let left = agent.pos + vec2f(cos(agent.angle - sensorAngle), sin(agent.angle - sensorAngle)) * sensorDist;
        let right = agent.pos + vec2f(cos(agent.angle + sensorAngle), sin(agent.angle + sensorAngle)) * sensorDist;
        
        let frontWrapped = vec2f(
            wrap(front.x, width),
            wrap(front.y, height)
        );
        
        let leftWrapped = vec2f(
            wrap(left.x, width),
            wrap(left.y, height)
        );
        
        let rightWrapped = vec2f(
            wrap(right.x, width),
            wrap(right.y, height)
        );
        
        // Sample trail + food values
        let trailF = textureLoad(trailTex, vec2i(frontWrapped), 0).x;
        let trailL = textureLoad(trailTex, vec2i(leftWrapped), 0).x;
        let trailR = textureLoad(trailTex, vec2i(rightWrapped), 0).x;

        let foodF = textureLoad(foodTex, vec2i(frontWrapped), 0).r;
        let foodL = textureLoad(foodTex, vec2i(leftWrapped), 0).r;
        let foodR = textureLoad(foodTex, vec2i(rightWrapped), 0).r;

        // Weighted attraction: more food = more turning
        let foodWeight = min(1.0, max(foodF, max(foodL, foodR)) * 5.0);
        let fVal = mix(trailF, foodF, foodWeight);
        let lVal = mix(trailL, foodL, foodWeight);
        let rVal = mix(trailR, foodR, foodWeight);
        
        // Update direction
        if (lVal > fVal && lVal > rVal) {
            agent.angle -= turnSpeed;
        } else if (rVal > fVal && rVal > lVal) {
            agent.angle += turnSpeed;
        } else if (fVal < lVal && fVal < rVal) {
            let rand = random(agent.pos.x + agent.pos.y + f32(i));
            if (rand > 0.5) {
                agent.angle += turnSpeed;
            } else {
                agent.angle -= turnSpeed;
            }
        }
        
        agent.pos += vec2f(cos(agent.angle), sin(agent.angle)) * moveSpeed;
        
        agent.pos.x = wrap(agent.pos.x, width);
        agent.pos.y = wrap(agent.pos.y, height);
              
        let pos = vec2i(agent.pos);
        textureStore(trailOut, pos, vec4f(1.0, 0.0, 0.0, 1.0));

        agents[i] = agent;
    }
`;

const DiffuseShader = `
    @group(0) @binding(0) var inputTex: texture_2d<f32>;
    @group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;
    @group(0) @binding(2) var<uniform> params: vec4f;
    @group(0) @binding(3) var<uniform> diffuseParams: vec2f;
    
    @compute @workgroup_size(16, 16)
    fn main(@builtin(global_invocation_id) id: vec3u) {
        let coord = vec2i(id.xy);
        let width = i32(params.x);
        let height = i32(params.y);
        
        if (coord.x >= width || coord.y >= height) { return; }
        
        let decay = diffuseParams.x;
        let diffuseRate = diffuseParams.y;
        
        let center = textureLoad(inputTex, coord, 0).x;
        
        var sum = 0.0;
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                var sampleCoord = coord + vec2i(dx, dy);
                
                if (sampleCoord.x < 0) { sampleCoord.x += width; }
                if (sampleCoord.x >= width) { sampleCoord.x -= width; }
                if (sampleCoord.y < 0) { sampleCoord.y += height; }
                if (sampleCoord.y >= height) { sampleCoord.y -= height; }
                
                sum += textureLoad(inputTex, sampleCoord, 0).x;
            }
        }
        
        let decayed = center * decay;
        let diffused = mix(decayed, sum / 9.0, diffuseRate);
        
        textureStore(outputTex, coord, vec4f(diffused, 0.0, 0.0, 1.0));
		
    }
`;

const destroyShader = `
	@group(0) @binding(0) var<uniform> params: vec3f;
	@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;

    @compute @workgroup_size(16, 16)
    fn main(@builtin(global_invocation_id) id: vec3u) {
		let coord = vec2f(id.xy);
		let center = params.xy;
		let radius = params.z;
		if (radius > 0.0 && distance(coord, center) < radius){
			textureStore(outputTex, vec2i(coord.xy), vec4f(0.0, 0.0, 0.0, 0.0));
		}
	}
`;

const destroyAgentsShader = `
    struct Agent {
        pos: vec2f,
        angle: f32,
    }

    @group(0) @binding(0) var<uniform> destroyParams: vec3f;
    @group(0) @binding(1) var<storage, read_write> agents: array<Agent>;
    @group(0) @binding(2) var<uniform> params: vec4f;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) id: vec3u) {
        let i = id.x;
        if (i >= arrayLength(&agents)) { return; }
        
        var agent = agents[i];
        let center = destroyParams.xy;
        let radius = destroyParams.z;
        
        if (radius > 0.0 && distance(agent.pos, center) < radius) {
            
            let width = params.x;
            let height = params.y;
            
            let seed = agent.pos.x + agent.pos.y + f32(i) * 0.1;
            
            let randX = fract(sin(seed * 78.233) * 43758.5453);
            let randY = fract(sin((seed + 1.0) * 78.233) * 43758.5453);
            let randAngle = fract(sin((seed + 2.0) * 78.233) * 43758.5453) * 6.28318;
            
            let edge = floor(randX * 4.0);
            
            if (edge < 1.0) {
                agent.pos = vec2f(randX * width, 0.0);
            } else if (edge < 2.0) {
                agent.pos = vec2f(width - 1.0, randY * height);
            } else if (edge < 3.0) {
                agent.pos = vec2f(randX * width, height - 1.0);
            } else {
                agent.pos = vec2f(0.0, randY * height);
            }
            
            agent.angle = randAngle;
            
            agents[i] = agent;
        }
    }
`;

const foodDecayShader = `
  @group(0) @binding(0) var inputTex: texture_2d<f32>;
  @group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;
  @group(0) @binding(2) var<uniform> params: vec4f;

  @compute @workgroup_size(16, 16)
  fn main(@builtin(global_invocation_id) id: vec3u) {
    let coord = vec2i(id.xy);
    let width = i32(params.x);
    let height = i32(params.y);
    if (coord.x >= width || coord.y >= height) { return; }

    let food = textureLoad(inputTex, coord, 0);

    // If alpha == 0, nothing to do
    if (food.a == 0.0) {
        textureStore(outputTex, coord, vec4f(0.0));
        return;
    }

    var r = max(0.0, food.r - 0.002);       // slower fade from red
    var g = min(0.15, food.g + 0.001);      // slower brown blending
    var b = min(0.03, food.b + 0.0005);     // barely changes
    var a = max(0.0, food.a - 0.002);       // slower alpha fade

    textureStore(outputTex, coord, vec4f(r, g, b, a));
  }
`;

async function start() {
  if (!navigator.gpu) {
    console.error("This browser does not support WebGPU");
    return;
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    console.error("This browser supports WebGPU but it appears disabled");
    return;
  }

  const device = await adapter?.requestDevice();
  device.lost.then((info) => {
    console.error(`WebGPU device was lost: ${info.message}`);

    if (info.reason !== "destroyed") {
      start();
    }
  });

  main(device);
}

async function main(device) {
  const gui = new GUI();

  const canvas = document.querySelector("canvas");
  if (!canvas) {
    canvas = document.createElement("canvas");
    document.body.appendChild(canvas);
  }

  const context = canvas.getContext("webgpu");
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

  context.configure({
    device: device,
    format: presentationFormat,
  });

  // Simulation parameters
  const WIDTH = (canvas.width = window.innerWidth);
  const HEIGHT = (canvas.height = window.innerHeight);
  const NUM_AGENTS = 250_000;

  var settings = {
    SENSOR_ANGLE: 0.3,
    SENSOR_DIST: 9,
    MOVE_SPEED: 1.0,
    TURN_SPEED: 0.1,
    DECAY: 0.98,
    DIFFUSE: 0.2,
    COLOR: [255, 255, 255],
    SHAPE: "random",
    RESET: function () {
      resetSimulation();
    },
  };

  // MAKE THIS RANGES MAKE SENSE
  gui.add(settings, "SENSOR_ANGLE", 0, 1);
  gui.add(settings, "SENSOR_DIST", 1, 15);
  gui.add(settings, "MOVE_SPEED", 1, 10);
  gui.add(settings, "TURN_SPEED", -10, 10);
  gui.add(settings, "DECAY", 0.1, 1);
  gui.add(settings, "DIFFUSE", 0.1, 1);
  gui.addColor(settings, "COLOR");
  gui.add(settings, "SHAPE", ["random", "circle"]);
  gui.add(settings, "RESET");

  function initAgents() {
    const agents = new Float32Array(NUM_AGENTS * 3);
    const centerX = WIDTH / 2;
    const centerY = HEIGHT / 2;
    switch (settings.SHAPE) {
      case "circle":
        const radius = Math.min(WIDTH, HEIGHT) / 3;
        for (let i = 0; i < NUM_AGENTS; i++) {
          const angle = Math.random() * Math.PI * 2;
          agents[i * 3] = centerX + Math.cos(angle) * radius;
          agents[i * 3 + 1] = centerY + Math.sin(angle) * radius;
          agents[i * 3 + 2] = angle;
        }
        break;
      default:
        for (let i = 0; i < NUM_AGENTS; i++) {
          agents[i * 3] = Math.random() * WIDTH;
          agents[i * 3 + 1] = Math.random() * HEIGHT;
          agents[i * 3 + 2] = Math.random() * Math.PI * 2;
        }
    }
    return agents;
  }

  let agents = initAgents();

  let agentBuffer = device.createBuffer({
    size: agents.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Float32Array(agentBuffer.getMappedRange()).set(agents);
  agentBuffer.unmap();

  // Trail textures (double buffered)
  let trailTextureDesc = {
    size: [WIDTH, HEIGHT],
    format: "rgba8unorm",
    usage:
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST,
  };

  const trailTextures = [
    device.createTexture(trailTextureDesc),
    device.createTexture(trailTextureDesc),
  ];

  function clearTextures() {
    const clearColor = new Uint8Array(WIDTH * HEIGHT * 4).fill(0);
    for (let i = 0; i < 2; i++) {
      device.queue.writeTexture(
        { texture: trailTextures[i] },
        clearColor,
        { bytesPerRow: WIDTH * 4, rowsPerImage: HEIGHT },
        [WIDTH, HEIGHT]
      );
    }
  }

  const foodTextures = [
    device.createTexture({
      size: [WIDTH, HEIGHT],
      format: "rgba8unorm",
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST,
    }),
    device.createTexture({
      size: [WIDTH, HEIGHT],
      format: "rgba8unorm",
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST,
    }),
  ];

  function drawFood(x, y, radius = 10) {
    const data = new Uint8Array(WIDTH * HEIGHT * 4).fill(0);

    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const nx = Math.floor(x + dx);
        const ny = Math.floor(y + dy);
        if (nx < 0 || ny < 0 || nx >= WIDTH || ny >= HEIGHT) continue;

        const dist = Math.hypot(dx, dy);
        if (dist <= radius) {
          const index = (ny * WIDTH + nx) * 4;
          data[index] = 255; // Red
          data[index + 1] = 0;
          data[index + 2] = 0;
          data[index + 3] = 255;
        }
      }
    }

    for (let tex of foodTextures) {
      device.queue.writeTexture(
        { texture: tex },
        data,
        { bytesPerRow: WIDTH * 4 },
        [WIDTH, HEIGHT]
      );
    }
  }

  clearTextures();

  // Uniform buffers
  var params = new Float32Array([WIDTH, HEIGHT, settings.COLOR, 0]);
  var paramBuffer = device.createBuffer({
    size: params.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(paramBuffer, 0, params);

  var color = new Float32Array([
    settings.COLOR[0],
    settings.COLOR[1],
    settings.COLOR[2],
  ]);
  var colorBuffer = device.createBuffer({
    size: params.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(colorBuffer, 0, color);

  var simParams = new Float32Array([
    settings.SENSOR_ANGLE,
    settings.SENSOR_DIST,
    settings.MOVE_SPEED,
    settings.TURN_SPEED,
  ]);
  var simParamBuffer = device.createBuffer({
    size: simParams.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(simParamBuffer, 0, simParams);

  var diffuseParams = new Float32Array([settings.DECAY, settings.DIFFUSE]);
  var diffuseParamBuffer = device.createBuffer({
    size: diffuseParams.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(diffuseParamBuffer, 0, diffuseParams);

  var destroy = new Float32Array([0, 0, -1]);
  var destroyBuffer = device.createBuffer({
    size: 12,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(destroyBuffer, 0, destroy);

  // Create shader modules
  const agentShaderModule = device.createShaderModule({ code: AgentShader });
  const diffuseShaderModule = device.createShaderModule({
    code: DiffuseShader,
  });
  const vertShaderModule = device.createShaderModule({ code: VertShader });
  const fragShaderModule = device.createShaderModule({ code: FragShader });
  const destroyShaderModule = device.createShaderModule({
    code: destroyShader,
  });
  const destroyAgentsShaderModule = device.createShaderModule({
    code: destroyAgentsShader,
  });
  const FoodDecayModule = device.createShaderModule({
    code: foodDecayShader,
  });

  // Compute pipelines
  const agentPipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: agentShaderModule,
      entryPoint: "main",
    },
  });

  const diffusePipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: diffuseShaderModule,
      entryPoint: "main",
    },
  });

  // Render pipeline
  const renderPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: vertShaderModule,
      entryPoint: "vs",
    },
    fragment: {
      module: fragShaderModule,
      entryPoint: "fs",
      targets: [{ format: presentationFormat }],
    },
  });

  // Destroy texture pipeline
  const destroyPipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: destroyShaderModule,
      entryPoint: "main",
    },
  });

  // Destroy agents pipeline
  const destroyAgentsPipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: destroyAgentsShaderModule,
      entryPoint: "main",
    },
  });

  // Create sampler for texture sampling in FS
  const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
    addressModeU: "clamp-to-edge",
    addressModeV: "clamp-to-edge",
  });

  const FoodDecayPipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: FoodDecayModule,
      entryPoint: "main",
    },
  });

  // Handle user input
  canvas.addEventListener("mousedown", (e) => {
    if (e.ctrlKey) { // Hold Ctrl to place food
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      const simX = (mouseX * WIDTH) / canvas.width;
      const simY = (mouseY * HEIGHT) / canvas.height;
      drawFood(simX, simY, 20);
      console.log("Food placed at:", simX, simY);
    }
    else{
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      const simX = (mouseX * WIDTH) / canvas.width;
      const simy = (mouseY * HEIGHT) / canvas.height;
      const radius = 150; // Maybe make this variable using GUI?
      destroy.set([simX, simy, radius]);
      device.queue.writeBuffer(destroyBuffer, 0, destroy);
      window.destroy = true;
      console.log(simX, simy, radius);
    }

  });

  window.addEventListener("keydown", (e) => {
    if (e.key.toLowerCase() === "f") {
      console.log("F key pressed");
      const x = Math.random() * WIDTH;
      const y = Math.random() * HEIGHT;
      drawFood(x, y, 10);
    }
  });

  let currentTexture = 0;
  let animationFrameId = null;

  let currentFood = 0;

  function resetSimulation() {
    if (animationFrameId) {
      cancelAnimationFrame(animationFrameId);
    }

    agents = initAgents();
    agentBuffer = device.createBuffer({
      size: agents.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(agentBuffer.getMappedRange()).set(agents);
    agentBuffer.unmap();

    clearAllTextures();
    currentTexture = 0;
    frame();
  }

  function clearAllTextures() {
    const clearColor = new Uint8Array(WIDTH * HEIGHT * 4).fill(0);
    for (let tex of [...trailTextures, ...foodTextures]) {
      device.queue.writeTexture(
        { texture: tex },
        clearColor,
        { bytesPerRow: WIDTH * 4, rowsPerImage: HEIGHT },
        [WIDTH, HEIGHT]
      );
    }
  }

  function frame() {
    const nextTexture = 1 - currentTexture;
    const encoder = device.createCommandEncoder();

    //Reset buffer params
    simParams.set([
      settings.SENSOR_ANGLE,
      settings.SENSOR_DIST,
      settings.MOVE_SPEED,
      settings.TURN_SPEED,
    ]);
    device.queue.writeBuffer(simParamBuffer, 0, simParams);

    diffuseParams.set([settings.DECAY, settings.DIFFUSE]);
    device.queue.writeBuffer(diffuseParamBuffer, 0, diffuseParams);

    color.set([settings.COLOR[0], settings.COLOR[1], settings.COLOR[2]]);
    device.queue.writeBuffer(colorBuffer, 0, color);

    // Agent update
    const agentPass = encoder.beginComputePass();
    agentPass.setPipeline(agentPipeline);
    agentPass.setBindGroup(
      0,
      device.createBindGroup({
        layout: agentPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: agentBuffer } },
          { binding: 1, resource: trailTextures[currentTexture].createView() },
          { binding: 2, resource: trailTextures[nextTexture].createView() },
          { binding: 3, resource: { buffer: paramBuffer } },
          { binding: 4, resource: { buffer: simParamBuffer } },
          { binding: 5, resource: foodTextures[currentFood].createView() }
        ],
      })
    );
    agentPass.dispatchWorkgroups(Math.ceil(NUM_AGENTS / 256));
    agentPass.end();

    // Diffuse pass
    const diffusePass = encoder.beginComputePass();
    diffusePass.setPipeline(diffusePipeline);
    diffusePass.setBindGroup(
      0,
      device.createBindGroup({
        layout: diffusePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: trailTextures[nextTexture].createView() },
          { binding: 1, resource: trailTextures[currentTexture].createView() },
          { binding: 2, resource: { buffer: paramBuffer } },
          { binding: 3, resource: { buffer: diffuseParamBuffer } },
        ],
      })
    );
    diffusePass.dispatchWorkgroups(
      Math.ceil(WIDTH / 16),
      Math.ceil(HEIGHT / 16)
    );
    diffusePass.end();

    const foodDecayPass = encoder.beginComputePass();
    foodDecayPass.setPipeline(FoodDecayPipeline);
    foodDecayPass.setBindGroup(0, device.createBindGroup({
      layout: FoodDecayPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: foodTextures[currentFood].createView() },
        { binding: 1, resource: foodTextures[1 - currentFood].createView() },
        { binding: 2, resource: { buffer: paramBuffer } },
      ],
    }));
    foodDecayPass.dispatchWorkgroups(Math.ceil(WIDTH / 16), Math.ceil(HEIGHT / 16));
    foodDecayPass.end();

    //Destroy pass
    if (window.destroy) {
      const destroyPass = encoder.beginComputePass();
      destroyPass.setPipeline(destroyPipeline);
      destroyPass.setBindGroup(
        0,
        device.createBindGroup({
          layout: destroyPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: destroyBuffer } },
            {
              binding: 1,
              resource: trailTextures[currentTexture].createView(),
            },
          ],
        })
      );
      destroyPass.dispatchWorkgroups(
        Math.ceil(WIDTH / 16),
        Math.ceil(HEIGHT / 16)
      );
      destroyPass.end();

      const destroyAgentsPass = encoder.beginComputePass();
      destroyAgentsPass.setPipeline(destroyAgentsPipeline);
      destroyAgentsPass.setBindGroup(
        0,
        device.createBindGroup({
          layout: destroyAgentsPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: destroyBuffer } },
            { binding: 1, resource: { buffer: agentBuffer } },
            { binding: 2, resource: { buffer: paramBuffer } },
          ],
        })
      );

      destroyAgentsPass.dispatchWorkgroups(Math.ceil(NUM_AGENTS / 256));
      destroyAgentsPass.end();

      window.destroy = false;
    }

    // Render pass
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: "clear",
          storeOp: "store",
          clearValue: [0, 0, 0, 1],
        },
      ],
    });
    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(
      0,
      device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: trailTextures[currentTexture].createView() },
          { binding: 1, resource: sampler },
          { binding: 2, resource: { buffer: paramBuffer } },
          { binding: 3, resource: { buffer: colorBuffer } },
          { binding: 4, resource: foodTextures[currentFood].createView() },
        ],
      })
    );
    renderPass.draw(3);
    renderPass.end();

    device.queue.submit([encoder.finish()]);
    currentTexture = nextTexture;
    currentFood = 1 - currentFood;
    animationFrameId = requestAnimationFrame(frame);
  }

  // Start animation
  frame();
}

start();
