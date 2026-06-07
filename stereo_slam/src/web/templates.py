"""HTML templates for the debug web server."""

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Stereo SLAM Web Visualization</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            background-color: #1a1a2e;
            color: #eee;
            font-family: Arial, sans-serif;
        }
        h1 {
            text-align: center;
            color: #00ff88;
        }
        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .video-box {
            text-align: center;
            margin-bottom: 20px;
        }
        .video-box h3 {
            margin: 5px 0;
            color: #00ff88;
        }
        img {
            border: 2px solid #333;
            border-radius: 8px;
            max-width: 100%;
        }
        .stats {
            background-color: #16213e;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            min-width: 300px;
        }
        .stats h3 {
            margin-top: 0;
            color: #00ff88;
        }
        .stat-item {
            display: flex;
            justify-content: space-between;
            gap: 16px;
            padding: 5px 0;
            border-bottom: 1px solid #333;
        }
        .stat-label {
            color: #aaa;
            flex: 0 0 auto;
        }
        .stat-value {
            color: #00ff88;
            font-weight: bold;
            max-width: 65%;
            overflow-wrap: anywhere;
            text-align: right;
        }
    </style>
</head>
<body>
    <h1>Stereo SLAM Real-time Visualization</h1>
    <div class="container">
        <div class="video-box">
            <h3>Camera (current stereo observations)</h3>
            <img id="video" src="" alt="Video Stream" width="640" height="360">
        </div>
        <div class="stats">
            <h3>Statistics</h3>
            <div class="stat-item">
                <span class="stat-label">Frame Count:</span>
                <span class="stat-value" id="frame">--</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Map Points:</span>
                <span class="stat-value" id="points">--</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Current Observations:</span>
                <span class="stat-value" id="observations">--</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Camera Position:</span>
                <span class="stat-value" id="campos">--</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Map Save:</span>
                <span class="stat-value" id="mapsave">--</span>
            </div>
        </div>
    </div>

    <script>
        function updateFrame() {
            document.getElementById('video').src = '/frame.jpg?t=' + Date.now();
        }

        function updateStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('frame').textContent = data.frame_count;
                    document.getElementById('points').textContent = data.num_points;
                    document.getElementById('observations').textContent =
                        data.num_current_observations || 0;
                    document.getElementById('campos').textContent =
                        '(' + data.camera_pos[0].toFixed(2) + ', ' +
                        data.camera_pos[1].toFixed(2) + ', ' +
                        data.camera_pos[2].toFixed(2) + ')';
                    if (data.map_save_path) {
                        const saveStatus = data.last_save_error
                            ? 'error: ' + data.last_save_error
                            : 'frame ' + (data.last_saved_frame_count || 0);
                        document.getElementById('mapsave').textContent =
                            saveStatus + ' -> ' + data.map_save_path;
                    } else {
                        document.getElementById('mapsave').textContent = 'disabled';
                    }
                })
                .catch(err => console.error('Error:', err));
        }

        setInterval(updateFrame, 50);
        setInterval(updateStats, 500);
        updateStats();
    </script>
</body>
</html>
"""
