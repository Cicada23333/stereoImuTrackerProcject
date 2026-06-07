"""HTML template for the read-only map localization web test."""

LOCALIZATION_INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Stereo Map Localization Test</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            background-color: #10131f;
            color: #eeeeee;
            font-family: Arial, sans-serif;
        }
        h1 {
            text-align: center;
            color: #20d6b5;
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
            color: #20d6b5;
        }
        img {
            border: 2px solid #333842;
            border-radius: 8px;
            max-width: 100%;
        }
        .stats {
            background-color: #171d2d;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            min-width: 360px;
            max-width: 760px;
        }
        .stats h3 {
            margin-top: 0;
            color: #20d6b5;
        }
        .stat-item {
            display: flex;
            justify-content: space-between;
            gap: 16px;
            padding: 5px 0;
            border-bottom: 1px solid #333842;
        }
        .stat-label {
            color: #a8b0bf;
            flex: 0 0 auto;
        }
        .stat-value {
            color: #20d6b5;
            font-weight: bold;
            max-width: 68%;
            overflow-wrap: anywhere;
            text-align: right;
        }
    </style>
</head>
<body>
    <h1>Stereo Map Localization Test</h1>
    <div class="container">
        <div class="video-box">
            <h3>Camera With Projected Map Points</h3>
            <img id="video" src="" alt="Video Stream" width="640" height="360">
        </div>
        <div class="stats">
            <h3>Statistics</h3>
            <div class="stat-item">
                <span class="stat-label">Frame Count:</span>
                <span class="stat-value" id="frame">--</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Status:</span>
                <span class="stat-value" id="status">--</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Map Points:</span>
                <span class="stat-value" id="points">--</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Described Points:</span>
                <span class="stat-value" id="described">--</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Map Matches:</span>
                <span class="stat-value" id="matches">--</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">PnP Used:</span>
                <span class="stat-value" id="used">--</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">PnP Inliers:</span>
                <span class="stat-value" id="inliers">--</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Inlier Ratio:</span>
                <span class="stat-value" id="ratio">--</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Median Error:</span>
                <span class="stat-value" id="mederr">--</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Camera Position:</span>
                <span class="stat-value" id="campos">--</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Map Path:</span>
                <span class="stat-value" id="mappath">--</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Error:</span>
                <span class="stat-value" id="error">--</span>
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
                    const result = data.last_result || {};
                    document.getElementById('frame').textContent = data.frame_count;
                    document.getElementById('status').textContent =
                        result.success ? 'localized (' + (result.quality || 'ok') + ')' : 'not localized';
                    document.getElementById('points').textContent = result.num_map_points || 0;
                    document.getElementById('described').textContent =
                        result.num_described_map_points || 0;
                    document.getElementById('matches').textContent = result.num_map_matches || 0;
                    document.getElementById('used').textContent = result.num_pnp_used_matches || 0;
                    document.getElementById('inliers').textContent = result.num_pnp_inliers || 0;
                    document.getElementById('ratio').textContent =
                        ((result.inlier_ratio || 0) * 100).toFixed(1) + '%';
                    document.getElementById('mederr').textContent =
                        result.median_inlier_reprojection_error == null
                            ? '--'
                            : result.median_inlier_reprojection_error.toFixed(2) + ' px';
                    const pos = result.camera_position || [0, 0, 0];
                    document.getElementById('campos').textContent =
                        '(' + pos[0].toFixed(2) + ', ' +
                        pos[1].toFixed(2) + ', ' +
                        pos[2].toFixed(2) + ')';
                    document.getElementById('mappath').textContent = data.map_path || '--';
                    document.getElementById('error').textContent = result.error || '--';
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
