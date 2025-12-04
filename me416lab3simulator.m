function multiRobotSimulator()
% MULTIROBOTSIMULATOR - Simulates 12 robots using dynamic path planning
% 
% This simulator tests the real-time path planning algorithm with 12 robots
% (1 "real" robot + 11 digital twins) all running the same controller code.
% Each robot has randomly assigned start and goal positions.
% OPEN SPACE - no static obstacles, robots only avoid each other
% PHYSICAL BOUNDARIES - Each robot has a 30cm x 40cm rectangular body

    clearvars; close all;
    rng(42); % Set random seed for reproducibility
    
    %% SIMULATION PARAMETERS
    SIM.numRobots = 12;
    SIM.maxTime = 300; % Maximum simulation time [s]
    SIM.dt = 1/30; % Time step [s] (30 Hz control rate)
    SIM.mapSize = [5.5, 5.0]; % [width, height] in meters
    SIM.realTimeMode = true; % Set to true for real-time visualization
    
    %% ROBOT PHYSICAL DIMENSIONS
    SIM.robotWidth = 0.30;   % 30 cm width (perpendicular to heading)
    SIM.robotLength = 0.40;  % 40 cm length (parallel to heading)
    
    %% CONTROLLER CONFIGURATION (same as real robot)
    CFG.lookaheadDist = 0.35;
    CFG.targetLinearVel = 0.25;
    CFG.minTurnRadius = 0.2;
    CFG.recoveryDuration = 1.0;
    CFG.posTolerance = 0.2;
    CFG.headingTolerance = deg2rad(25);
    CFG.maxLinVel = 0.25;
    CFG.maxAngVel = deg2rad(60);
    
    % Path planning parameters
    CFG.replanInterval = 1.0;
    CFG.rrtMaxIter = 500;
    CFG.rrtStepSize = 0.3;
    CFG.rrtGoalBias = 0.15;
    CFG.rrtNeighborRadius = 0.8;
    
    % Dynamic obstacle parameters (adjusted for rectangular bodies)
    CFG.robotSafetyMargin = 0.15;     % Extra safety margin around robot body [m]
    CFG.predictionHorizon = 3.0;      % How far ahead to predict robot motion [s]
    CFG.velocityScaleFactor = 2.0;    % Scale velocity for elongated keepout zones
    CFG.obstacleInflation = 2.0;      % Inflate obstacles for safer planning
    
    % Kalman filter parameters
    CFG.kf_processNoise = 0.1;
    CFG.kf_measureNoise = 0.05;
    
    %% NO STATIC OBSTACLES - Open space navigation
    staticObstacles = []; % Empty - robots only avoid each other
    
    %% INITIALIZE ROBOTS
    fprintf('Initializing %d robots with 30cm x 40cm bodies in open space...\n', SIM.numRobots);
    robots = struct();
    
    % Define margin from edges (account for robot size)
    edgeMargin = 0.5;
    
    for i = 1:SIM.numRobots
        robotId = sprintf('robot%02d', i);
        
        % Generate random start position (with minimum separation)
        validStart = false;
        attempts = 0;
        while ~validStart && attempts < 100
            x = edgeMargin + rand * (SIM.mapSize(1) - 2*edgeMargin);
            y = edgeMargin + rand * (SIM.mapSize(2) - 2*edgeMargin);
            theta = rand * 2 * pi; % Random initial heading
            
            % Check distance to other robots (use conservative circular approximation for initialization)
            validStart = true;
            existingIds = fieldnames(robots);
            for j = 1:length(existingIds)
                dist = sqrt((x - robots.(existingIds{j}).pose(1))^2 + ...
                           (y - robots.(existingIds{j}).pose(2))^2);
                if dist < 1.0  % Conservative separation for initialization
                    validStart = false;
                    break;
                end
            end
            attempts = attempts + 1;
        end
        
        % Generate random goal position (different from start)
        validGoal = false;
        attempts = 0;
        while ~validGoal && attempts < 100
            gx = edgeMargin + rand * (SIM.mapSize(1) - 2*edgeMargin);
            gy = edgeMargin + rand * (SIM.mapSize(2) - 2*edgeMargin);
            
            % Must be far from start
            distFromStart = sqrt((gx - x)^2 + (gy - y)^2);
            if distFromStart > 2.0
                validGoal = true;
            end
            attempts = attempts + 1;
        end
        
        % Initialize robot state
        robots.(robotId).pose = [x, y, theta]; % [x, y, theta]
        robots.(robotId).velocity = [0, 0]; % [vx, vy]
        robots.(robotId).goal = [gx; gy];
        robots.(robotId).trajectory = [x, y];
        robots.(robotId).reached = false;
        robots.(robotId).reachedTime = inf;
        robots.(robotId).lastReplanTime = -inf;
        robots.(robotId).replanCount = 0;
        robots.(robotId).recoveryTimer = [];
        robots.(robotId).errorHistory = [];
        
        % Initialize with straight-line path to goal
        nPts = 20;
        robots.(robotId).path.x = linspace(x, gx, nPts)';
        robots.(robotId).path.y = linspace(y, gy, nPts)';
        robots.(robotId).path.s = [0; cumsum(sqrt(diff(robots.(robotId).path.x).^2 + ...
                                                   diff(robots.(robotId).path.y).^2))];
        
        % Kalman filter tracking for this robot (from perspective of others)
        robots.(robotId).kf_state = [x; y; 0; 0]; % [x, y, vx, vy]
        robots.(robotId).kf_P = eye(4) * 0.1;
        Q = eye(4) * CFG.kf_processNoise;
        Q(1:2,1:2) = Q(1:2,1:2) * 0.1;
        robots.(robotId).kf_Q = Q;
        robots.(robotId).kf_R = eye(2) * CFG.kf_measureNoise;
        
        fprintf('  %s: Start(%.2f, %.2f, %.0f°) -> Goal(%.2f, %.2f)\n', ...
                robotId, x, y, rad2deg(theta), gx, gy);
    end
    
    %% VISUALIZATION SETUP
    figure('Name', 'Multi-Robot Simulation - Live View with Physical Bodies', ...
           'Position', [50 50 1400 900]);
    
    % Color map for robots
    colors = lines(SIM.numRobots);
    
    % Initialize plots for each robot
    robotIds = fieldnames(robots);
    plotHandles = struct();
    
    hold on; grid on; axis equal;
    
    for i = 1:length(robotIds)
        rid = robotIds{i};
        col = colors(i,:);
        
        % Goal marker
        plot(robots.(rid).goal(1), robots.(rid).goal(2), 'x', ...
             'Color', col, 'MarkerSize', 15, 'LineWidth', 2);
        
        % Start marker
        plot(robots.(rid).pose(1), robots.(rid).pose(2), '*', ...
             'Color', col, 'MarkerSize', 10, 'LineWidth', 1.5);
        
        % Path line
        plotHandles.(rid).path = plot(robots.(rid).path.x, robots.(rid).path.y, ...
                                      '--', 'Color', col, 'LineWidth', 1.0, 'Color', [col 0.3]);
        
        % Trajectory line
        plotHandles.(rid).traj = plot(NaN, NaN, ':', 'Color', col, 'LineWidth', 2.0);
        
        % Robot center position (small dot)
        plotHandles.(rid).robot = plot(NaN, NaN, 'o', 'Color', col, ...
                                       'MarkerSize', 6, 'MarkerFaceColor', col);
        
        % Robot body (30cm x 40cm rectangle)
        plotHandles.(rid).body = patch('XData', NaN, 'YData', NaN, ...
                                       'FaceColor', col, 'FaceAlpha', 0.5, ...
                                       'EdgeColor', col, 'LineWidth', 2);
        
        % Robot heading indicator (arrow)
        plotHandles.(rid).heading = plot(NaN, NaN, '-', 'Color', col, 'LineWidth', 3);
    end
    
    xlabel('X Position [m]'); ylabel('Y Position [m]');
    title('Multi-Robot Simulation - Real-Time Path Planning with Physical Bodies (30cm x 40cm)');
    xlim([-0.5, SIM.mapSize(1)]); ylim([-0.5, SIM.mapSize(2)]);
    
    %% SIMULATION LOOP
    if SIM.realTimeMode
        fprintf('\nStarting REAL-TIME simulation...\n');
        fprintf('Watch the robots navigate with physical body constraints!\n\n');
    else
        fprintf('\nStarting FAST simulation...\n\n');
    end
    
    simTime = 0;
    frameCount = 0;
    
    while simTime < SIM.maxTime
        frameStart = tic;
        
        % Update all Kalman filters with current measurements
        for i = 1:length(robotIds)
            rid = robotIds{i};
            if robots.(rid).reached
                continue;
            end
            
            % Kalman prediction
            F = [1 0 SIM.dt 0;
                 0 1 0 SIM.dt;
                 0 0 1 0;
                 0 0 0 1];
            robots.(rid).kf_state = F * robots.(rid).kf_state;
            robots.(rid).kf_P = F * robots.(rid).kf_P * F' + robots.(rid).kf_Q;
            
            % Kalman update
            H = [1 0 0 0; 0 1 0 0];
            z = robots.(rid).pose(1:2)';
            y_inn = z - H * robots.(rid).kf_state;
            S = H * robots.(rid).kf_P * H' + robots.(rid).kf_R;
            K = robots.(rid).kf_P * H' / S;
            robots.(rid).kf_state = robots.(rid).kf_state + K * y_inn;
            robots.(rid).kf_P = (eye(4) - K * H) * robots.(rid).kf_P;
            
            % Update velocity estimate
            robots.(rid).velocity = robots.(rid).kf_state(3:4)';
        end
        
        % Control and update each robot
        for i = 1:length(robotIds)
            rid = robotIds{i};
            
            if robots.(rid).reached
                continue;
            end
            
            % Check if goal reached
            distToGoal = sqrt((robots.(rid).pose(1) - robots.(rid).goal(1))^2 + ...
                             (robots.(rid).pose(2) - robots.(rid).goal(2))^2);
            if distToGoal < CFG.posTolerance
                robots.(rid).reached = true;
                robots.(rid).reachedTime = simTime;
                fprintf('  [%.1fs] %s reached goal (replans: %d)\n', ...
                        simTime, rid, robots.(rid).replanCount);
                continue;
            end
            
            % Path replanning
            if (simTime - robots.(rid).lastReplanTime) >= CFG.replanInterval
                % Create dynamic obstacles from other robots (with rectangular bodies)
                otherRobotsList = createOtherRobotsList(robots, rid, CFG);
                obstacles = createDynamicObstaclesSim(staticObstacles, otherRobotsList, ...
                                                     CFG, SIM.robotWidth, SIM.robotLength);
                
                % Run RRT* planner
                newPath = rrtStarPlanner(robots.(rid).pose, robots.(rid).goal, obstacles, ...
                                        CFG, SIM.mapSize, SIM.robotWidth, SIM.robotLength);
                
                if ~isempty(newPath)
                    robots.(rid).path = smoothPathSim(newPath, CFG.minTurnRadius);
                    robots.(rid).replanCount = robots.(rid).replanCount + 1;
                    
                    % Update path visualization
                    set(plotHandles.(rid).path, 'XData', robots.(rid).path.x, ...
                        'YData', robots.(rid).path.y);
                end
                robots.(rid).lastReplanTime = simTime;
            end
            
            % Pure Pursuit Control
            [lookaheadX, lookaheadY, ~, crossTrackError] = ...
                findLookaheadSim(robots.(rid).pose(1), robots.(rid).pose(2), ...
                                 robots.(rid).path, CFG.lookaheadDist);
            
            % Update error history
            robots.(rid).errorHistory = [robots.(rid).errorHistory, crossTrackError];
            if length(robots.(rid).errorHistory) > 30
                robots.(rid).errorHistory = robots.(rid).errorHistory(end-29:end);
            end
            
            % Recovery mode
            if ~isempty(robots.(rid).recoveryTimer)
                recoveryElapsed = simTime - robots.(rid).recoveryTimer;
                if recoveryElapsed < CFG.recoveryDuration
                    linVel = 0.0;
                    angVel = deg2rad(30);
                else
                    robots.(rid).recoveryTimer = [];
                end
            else
                % Normal Pure Pursuit control
                dx = lookaheadX - robots.(rid).pose(1);
                dy = lookaheadY - robots.(rid).pose(2);
                desiredHeading = atan2(dy, dx);
                headingError = wrapToPi(desiredHeading - robots.(rid).pose(3));
                distToLookahead = sqrt(dx^2 + dy^2);
                
                angVel = (2 * CFG.targetLinearVel * sin(headingError)) / distToLookahead;
                
                if abs(headingError) < deg2rad(90)
                    linVel = CFG.targetLinearVel * cos(headingError);
                else
                    linVel = CFG.targetLinearVel * 0.3;
                end
            end
            
            % Apply velocity limits
            linVel = max(0, min(linVel, CFG.maxLinVel));
            angVel = max(-CFG.maxAngVel, min(angVel, CFG.maxAngVel));
            
            % Update robot state (simple kinematic model)
            robots.(rid).pose(1) = robots.(rid).pose(1) + linVel * cos(robots.(rid).pose(3)) * SIM.dt;
            robots.(rid).pose(2) = robots.(rid).pose(2) + linVel * sin(robots.(rid).pose(3)) * SIM.dt;
            robots.(rid).pose(3) = wrapToPi(robots.(rid).pose(3) + angVel * SIM.dt);
            
            % Update trajectory
            robots.(rid).trajectory = [robots.(rid).trajectory; robots.(rid).pose(1:2)];
        end
        
        % Update visualization EVERY FRAME in real-time mode
        if SIM.realTimeMode || mod(frameCount, 5) == 0
            for i = 1:length(robotIds)
                rid = robotIds{i};
                
                x = robots.(rid).pose(1);
                y = robots.(rid).pose(2);
                theta = robots.(rid).pose(3);
                
                % Update robot center position
                set(plotHandles.(rid).robot, 'XData', x, 'YData', y);
                
                % Calculate rectangle corners (30cm width x 40cm length)
                % Rectangle centered at robot position
                halfWidth = SIM.robotWidth / 2;
                halfLength = SIM.robotLength / 2;
                
                % Local coordinates (before rotation)
                local_corners = [-halfLength, -halfWidth;   % Back left
                                 halfLength,  -halfWidth;   % Front left
                                 halfLength,   halfWidth;   % Front right
                                 -halfLength,  halfWidth];  % Back right
                
                % Rotation matrix
                R = [cos(theta), -sin(theta);
                     sin(theta),  cos(theta)];
                
                % Rotate and translate corners
                global_corners = (R * local_corners')';
                rectX = global_corners(:,1) + x;
                rectY = global_corners(:,2) + y;
                
                % Update rectangle body
                set(plotHandles.(rid).body, 'XData', rectX, 'YData', rectY);
                
                % Update heading indicator (arrow from center to front)
                headingLen = SIM.robotLength / 2 + 0.1;
                hx = [x, x + headingLen * cos(theta)];
                hy = [y, y + headingLen * sin(theta)];
                set(plotHandles.(rid).heading, 'XData', hx, 'YData', hy);
                
                % Update trajectory
                set(plotHandles.(rid).traj, 'XData', robots.(rid).trajectory(:,1), ...
                    'YData', robots.(rid).trajectory(:,2));
            end
            
            % Count reached robots properly
            numReached = 0;
            for i = 1:length(robotIds)
                if robots.(robotIds{i}).reached
                    numReached = numReached + 1;
                end
            end
            
            % Update title with stats
            title(sprintf('Multi-Robot Simulation | t=%.1fs | Robots: %d/%d reached goal | Bodies: 30cm x 40cm', ...
                         simTime, numReached, SIM.numRobots));
            drawnow limitrate;
        end
        
        % Check if all robots reached goal
        allReached = true;
        for i = 1:length(robotIds)
            if ~robots.(robotIds{i}).reached
                allReached = false;
                break;
            end
        end
        
        if allReached
            fprintf('\n✓ All robots reached their goals!\n');
            break;
        end
        
        % Advance simulation time
        simTime = simTime + SIM.dt;
        frameCount = frameCount + 1;
        
        % Maintain real-time rate (if enabled)
        if SIM.realTimeMode
            elapsed = toc(frameStart);
            if elapsed < SIM.dt
                pause(SIM.dt - elapsed);
            end
        end
    end
    
    %% SIMULATION RESULTS
    fprintf('\n=== Simulation Complete ===\n');
    fprintf('Total time: %.1f seconds\n', simTime);
    
    for i = 1:length(robotIds)
        rid = robotIds{i};
        if robots.(rid).reached
            fprintf('  %s: Reached in %.1fs (%d replans)\n', ...
                    rid, robots.(rid).reachedTime, robots.(rid).replanCount);
        else
            fprintf('  %s: Did not reach goal\n', rid);
        end
    end
    
    fprintf('\nSimulation finished.\n');
end

%% HELPER: CREATE OTHER ROBOTS LIST
function otherRobotsList = createOtherRobotsList(robots, myRobotId, CFG)
    otherRobotsList = [];
    robotIds = fieldnames(robots);
    
    for i = 1:length(robotIds)
        rid = robotIds{i};
        if strcmp(rid, myRobotId) || robots.(rid).reached
            continue;
        end
        
        otherRobot.x = robots.(rid).kf_state(1);
        otherRobot.y = robots.(rid).kf_state(2);
        otherRobot.vx = robots.(rid).kf_state(3);
        otherRobot.vy = robots.(rid).kf_state(4);
        otherRobot.theta = robots.(rid).pose(3); % Include heading for rectangular body
        
        otherRobotsList = [otherRobotsList; otherRobot];
    end
end

%% HELPER: CREATE DYNAMIC OBSTACLES (SIMULATOR VERSION WITH RECTANGULAR BODIES)
function obstacles = createDynamicObstaclesSim(staticObstacles, otherRobotsList, CFG, robotWidth, robotLength)
    obstacles = [];
    obstacleCount = 0;
    
    % Add static obstacles (none in open space mode)
    for i = 1:size(staticObstacles, 2)
        obstacleCount = obstacleCount + 1;
        obstacles(obstacleCount).x = staticObstacles(1, i);
        obstacles(obstacleCount).y = staticObstacles(2, i);
        obstacles(obstacleCount).radius = 0.15;
        obstacles(obstacleCount).type = 'circle';
    end
    
    % Add dynamic robot obstacles as grid of circles covering the rectangular body
    for i = 1:length(otherRobotsList)
        robot = otherRobotsList(i);
        speed = sqrt(robot.vx^2 + robot.vy^2);
        
        % Create a dense grid of circular obstacles to fully cover the rectangle
        % This ensures no part of the rectangle can be penetrated
        numCirclesLength = 7; % More circles along length for better coverage
        numCirclesWidth = 5;  % Circles along width
        
        % Base radius sized to ensure overlap between circles
        baseRadius = max(robotWidth, robotLength) / 6 + CFG.robotSafetyMargin;
        
        % Cover the entire rectangular footprint with overlapping circles
        for jL = 0:numCirclesLength-1
            for jW = 0:numCirclesWidth-1
                % Position in local robot frame
                offsetLength = (jL / (numCirclesLength-1) - 0.5) * robotLength;
                offsetWidth = (jW / (numCirclesWidth-1) - 0.5) * robotWidth;
                
                % Transform to global frame
                localX = offsetLength;
                localY = offsetWidth;
                globalX = robot.x + localX * cos(robot.theta) - localY * sin(robot.theta);
                globalY = robot.y + localX * sin(robot.theta) + localY * cos(robot.theta);
                
                obstacleCount = obstacleCount + 1;
                obstacles(obstacleCount).x = globalX;
                obstacles(obstacleCount).y = globalY;
                obstacles(obstacleCount).radius = baseRadius * CFG.obstacleInflation;
                obstacles(obstacleCount).type = 'circle';
            end
        end
        
        % Add predicted positions with full body coverage
        if speed > 0.02
            numPredictions = 4;
            for j = 1:numPredictions
                predTime = (j / numPredictions) * CFG.predictionHorizon;
                predX = robot.x + robot.vx * predTime;
                predY = robot.y + robot.vy * predTime;
                % Predict heading change based on angular velocity (approximate)
                predTheta = robot.theta;
                
                % Add grid of circles for predicted position
                for jL = 0:numCirclesLength-1
                    for jW = 0:numCirclesWidth-1
                        offsetLength = (jL / (numCirclesLength-1) - 0.5) * robotLength;
                        offsetWidth = (jW / (numCirclesWidth-1) - 0.5) * robotWidth;
                        
                        localX = offsetLength;
                        localY = offsetWidth;
                        globalX = predX + localX * cos(predTheta) - localY * sin(predTheta);
                        globalY = predY + localX * sin(predTheta) + localY * cos(predTheta);
                        
                        obstacleCount = obstacleCount + 1;
                        obstacles(obstacleCount).x = globalX;
                        obstacles(obstacleCount).y = globalY;
                        % Increase radius for predictions (more uncertainty)
                        obstacles(obstacleCount).radius = baseRadius * CFG.obstacleInflation * (1 + 0.4 * j/numPredictions);
                        obstacles(obstacleCount).type = 'circle';
                    end
                end
            end
        end
    end
end

%% HELPER: RRT* PLANNER (SIMULATOR VERSION WITH RECTANGULAR ROBOT)
function path = rrtStarPlanner(startPose, goalPos, obstacles, CFG, mapSize, robotWidth, robotLength)
    tree(1).x = startPose(1);
    tree(1).y = startPose(2);
    tree(1).theta = startPose(3);
    tree(1).parent = 0;
    tree(1).cost = 0;
    
    goalX = goalPos(1);
    goalY = goalPos(2);
    
    xMin = 0; xMax = mapSize(1);
    yMin = 0; yMax = mapSize(2);
    
    % Robot collision radius - use diagonal of rectangle plus safety margin
    % This ensures the entire robot body is safe when center follows path
    robotRadius = sqrt((robotWidth/2)^2 + (robotLength/2)^2) + CFG.robotSafetyMargin * 2;
    
    for iter = 1:CFG.rrtMaxIter
        if rand < CFG.rrtGoalBias
            xRand = goalX;
            yRand = goalY;
        else
            xRand = xMin + rand * (xMax - xMin);
            yRand = yMin + rand * (yMax - yMin);
        end
        
        distances = sqrt(([tree.x] - xRand).^2 + ([tree.y] - yRand).^2);
        [~, nearestIdx] = min(distances);
        xNearest = tree(nearestIdx).x;
        yNearest = tree(nearestIdx).y;
        
        angle = atan2(yRand - yNearest, xRand - xNearest);
        xNew = xNearest + CFG.rrtStepSize * cos(angle);
        yNew = yNearest + CFG.rrtStepSize * sin(angle);
        
        if ~isCollisionFreeSim(xNearest, yNearest, xNew, yNew, obstacles, robotRadius)
            continue;
        end
        
        distances = sqrt(([tree.x] - xNew).^2 + ([tree.y] - yNew).^2);
        nearInds = find(distances < CFG.rrtNeighborRadius);
        
        minCost = inf;
        bestParent = nearestIdx;
        for i = 1:length(nearInds)
            idx = nearInds(i);
            cost = tree(idx).cost + sqrt((tree(idx).x - xNew)^2 + (tree(idx).y - yNew)^2);
            if cost < minCost && isCollisionFreeSim(tree(idx).x, tree(idx).y, xNew, yNew, obstacles, robotRadius)
                minCost = cost;
                bestParent = idx;
            end
        end
        
        newIdx = length(tree) + 1;
        tree(newIdx).x = xNew;
        tree(newIdx).y = yNew;
        tree(newIdx).theta = atan2(yNew - tree(bestParent).y, xNew - tree(bestParent).x);
        tree(newIdx).parent = bestParent;
        tree(newIdx).cost = minCost;
        
        for i = 1:length(nearInds)
            idx = nearInds(i);
            cost = tree(newIdx).cost + sqrt((tree(idx).x - xNew)^2 + (tree(idx).y - yNew)^2);
            if cost < tree(idx).cost && isCollisionFreeSim(xNew, yNew, tree(idx).x, tree(idx).y, obstacles, robotRadius)
                tree(idx).parent = newIdx;
                tree(idx).cost = cost;
            end
        end
        
        distToGoal = sqrt((xNew - goalX)^2 + (yNew - goalY)^2);
        if distToGoal < CFG.rrtStepSize
            path = extractPathSim(tree, newIdx);
            return;
        end
    end
    
    distances = sqrt(([tree.x] - goalX).^2 + ([tree.y] - goalY).^2);
    [minDist, closestIdx] = min(distances);
    if minDist < 1.5  % Increased tolerance
        path = extractPathSim(tree, closestIdx);
    else
        path = [];
    end
end

%% HELPER: COLLISION CHECK WITH ROBOT RADIUS
function isFree = isCollisionFreeSim(x1, y1, x2, y2, obstacles, robotRadius)
    isFree = true;
    numChecks = 10;
    for i = 0:numChecks
        t = i / numChecks;
        x = x1 + t * (x2 - x1);
        y = y1 + t * (y2 - y1);
        
        for j = 1:length(obstacles)
            dist = sqrt((x - obstacles(j).x)^2 + (y - obstacles(j).y)^2);
            if dist < (obstacles(j).radius + robotRadius)
                isFree = false;
                return;
            end
        end
    end
end

%% HELPER: EXTRACT PATH
function path = extractPathSim(tree, goalIdx)
    pathX = [];
    pathY = [];
    currentIdx = goalIdx;
    
    while currentIdx ~= 0
        pathX = [tree(currentIdx).x; pathX];
        pathY = [tree(currentIdx).y; pathY];
        currentIdx = tree(currentIdx).parent;
    end
    
    path.x = pathX;
    path.y = pathY;
    path.s = [0; cumsum(sqrt(diff(pathX).^2 + diff(pathY).^2))];
end

%% HELPER: SMOOTH PATH
function smoothPath = smoothPathSim(roughPath, minTurnRadius)
    if length(roughPath.x) < 3
        smoothPath = roughPath;
        return;
    end
    
    numWaypoints = min(length(roughPath.x), 8);
    indices = round(linspace(1, length(roughPath.x), numWaypoints));
    wpX = roughPath.x(indices);
    wpY = roughPath.y(indices);
    
    waypoints = zeros(numWaypoints, 3);
    for i = 1:numWaypoints-1
        dx = wpX(i+1) - wpX(i);
        dy = wpY(i+1) - wpY(i);
        waypoints(i,:) = [wpX(i), wpY(i), atan2(dy, dx)];
    end
    waypoints(numWaypoints,:) = [wpX(numWaypoints), wpY(numWaypoints), waypoints(numWaypoints-1,3)];
    
    dubinsPlanner = dubinsConnection('MinTurningRadius', minTurnRadius);
    fullPath = [];
    
    for i = 1:numWaypoints-1
        try
            [segmentObject,~] = connect(dubinsPlanner, waypoints(i,:), waypoints(i+1,:));
            if ~isempty(segmentObject)
                segmentPoints = interpolate(segmentObject{1}, 0:0.05:segmentObject{1}.Length);
                if i > 1
                    segmentPoints = segmentPoints(2:end,:);
                end
                fullPath = [fullPath; segmentPoints];
            end
        catch
            nPts = 20;
            xSeg = linspace(waypoints(i,1), waypoints(i+1,1), nPts)';
            ySeg = linspace(waypoints(i,2), waypoints(i+1,2), nPts)';
            thetaSeg = ones(nPts,1) * waypoints(i,3);
            fullPath = [fullPath; xSeg ySeg thetaSeg];
        end
    end
    
    if isempty(fullPath)
        smoothPath = roughPath;
    else
        smoothPath.x = fullPath(:,1);
        smoothPath.y = fullPath(:,2);
        smoothPath.s = [0; cumsum(sqrt(diff(smoothPath.x).^2 + diff(smoothPath.y).^2))];
    end
end

%% HELPER: FIND LOOKAHEAD
function [lookaheadX, lookaheadY, lookaheadIndex, crossTrackError] = ...
    findLookaheadSim(robotX, robotY, plannedPath, lookaheadDistance)
    
    allDistances = sqrt((plannedPath.x - robotX).^2 + (plannedPath.y - robotY).^2);
    [crossTrackError, closestIndex] = min(allDistances);
    
    sClosest = plannedPath.s(closestIndex);
    sLookaheadTarget = sClosest + lookaheadDistance;
    
    if sLookaheadTarget > plannedPath.s(end)
        lookaheadIndex = length(plannedPath.x);
    else
        lookaheadIndex = find(plannedPath.s >= sLookaheadTarget, 1, 'first');
        if isempty(lookaheadIndex)
            lookaheadIndex = length(plannedPath.x);
        end
    end
    
    lookaheadX = plannedPath.x(lookaheadIndex);
    lookaheadY = plannedPath.y(lookaheadIndex);
end
