//TODO: perform error check

/*
** Road Network Definition
*/
INTERSECTION_COLOR = 'grey'
LANE_BORDER_COLOR = 'black'
LANE_BORDER_WIDTH = 1
LANE_INNER_COLOR = 'grey'
LANE_DASH_ARRAY = [10, 12]
TRAFFIC_LIGHT_WIDTH = 3

CAR_SIZE = 2
CAR_COLOR = 'blue'

var roadLayer = project.activeLayer
var carLayer = new Layer()
// define a symbol for car
car = new Path.RegularPolygon([0, 0], 3, CAR_SIZE)
car.fillColor = CAR_COLOR
car.rotate(90)
carSymbol = new Symbol(car)
roadLayer.activate()

nodes = {}
edges = {}
trafficLights = {}

simulation = null

jsonFile = 'test.json'
$.getJSON(jsonFile, function(json) {
    simulation = json

    for (i = 0;i < simulation.static.nodes.length;++i) {
        node = simulation.static.nodes[i]
        node.point = new Point(node.point)
        node.intersectConvex = {}
        nodes[node.id] = node
    }

    for (i = 0;i < simulation.static.edges.length;++i) {
        edge = simulation.static.edges[i]
        edge.from = nodes[edge.from]
        edge.to = nodes[edge.to]
        for (j = 0;j < edge.points.length;++j) {
            edge.points[j] = new Point(edge.points[j])
        }
        edges[edge.id] = edge
    }
    
    view.center = [200, 300]
    for (edgeId in edges) {
        drawEdge(edges[edgeId])
    }

    for (nodeId in nodes) {
        drawNode(nodes[nodeId])
    }

    roadLayer.scale(1, -1)
    totalStep = simulation.dynamic.length
})

lastTime = 0
step = 0
totalStep = 0
function onFrame(event) {
    //if (event.time - lastTime >= 0.01 && simulation && simulation.dynamic) {
    if (simulation && simulation.dynamic) {
        $("#frameRate").text(Math.round(1 / (event.time - lastTime)))
        drawStep(step)
        step += 1
        if (step >= totalStep) step = 0
        lastTime = event.time
    }
}

/*
** Helper Functions
*/
ROTATE = -90
function drawEdge(edge) {
    from = edge.from
    to = edge.to
    //points = [].concat(from.point, edge.points, to.point)
    points = edge.points

    prevPointBOffset = null
    for (i = 1;i < points.length;++i) {
        if (i == 1) {
            pointA = points[i-1] + (points[i] - points[i-1]).normalize() * (from.virtual ? 0 : from.width)
            pointAOffset = (points[i] - points[i-1]).rotate(ROTATE).normalize()
        } else {
            pointA = points[i-1]
            pointAOffset = prevPointBOffset
        }
        if (i == points.length - 1) {
            pointB = points[i] - (points[i] - points[i-1]).normalize() * (to.virtual ? 0 : to.width)
            pointBOffset = (points[i] - points[i - 1]).rotate(ROTATE).normalize()
        } else {
            pointB = points[i]
            pointBOffset = (points[i + 1] - points[i - 1]).rotate(ROTATE).normalize()
        }
        prevPointBOffset = pointBOffset

        // Draw Traffic Lights
        if (i == points.length-1 && !to.virtual) {
            edgeTrafficLights = []
            prevOffset = 0
            offset = 0
            for (lane = 0;lane < edge.nLane;++lane) {
                offset += edge.laneWidths[lane]
                path = new Path()
                path.strokeColor = 'grey'
                path.strokeWidth = TRAFFIC_LIGHT_WIDTH
                path.add(pointB + pointBOffset * prevOffset, pointB + pointBOffset * offset)
                edgeTrafficLights.push(path)
                prevOffset = offset
            }
            trafficLights[edge.id] = edgeTrafficLights
        }
         
        // Draw Roads
        path = new Path()
        path.strokeColor = LANE_BORDER_COLOR
        path.strokeWidth = LANE_BORDER_WIDTH
        path.add(pointA, pointB)

        offset = 0
        for (lane = 0;lane < edge.nLane-1;++lane) {
            offset += edge.laneWidths[lane]
            path = new Path()
            path.strokeColor = LANE_INNER_COLOR
            path.dashArray = LANE_DASH_ARRAY
            path.add(pointA + pointAOffset * offset, pointB + pointBOffset * offset)
        }

        offset += edge.laneWidths[edge.nLane-1]
        path = new Path()
        path.strokeColor = LANE_BORDER_COLOR
        path.strokeWidth = LANE_BORDER_WIDTH
        path.add(pointA + pointAOffset * offset, pointB + pointBOffset * offset)

        // Record Intersection Points
        if (i == 1) {
            if (from.intersectConvex[to.id] == null)
                from.intersectConvex[to.id] = []
            from.intersectConvex[to.id].push(pointA)
            from.intersectConvex[to.id].push(pointA + pointAOffset * offset)
        }
        if (i == points.length-1) {
            if (to.intersectConvex[from.id] == null)
                to.intersectConvex[from.id] = []
            to.intersectConvex[from.id].push(pointB)
            to.intersectConvex[from.id].push(pointB + pointBOffset * offset)
        }
    }
}

function drawNode(node) {
    node.intersectConvexPoint = []
    node.intersectConvexDegree = []
    
    nNode = 0
    for (nodeId in node.intersectConvex) nNode++
    if (nNode == 1) return

    for (nodeId in node.intersectConvex) {
        nodePoints = node.intersectConvex[nodeId]
        dir = nodes[nodeId].point - node.point
        minAngle = 360
        maxAngle = -360
        left = null
        right = null
        for (i = 0;i < nodePoints.length;++i) {
            angle = dir.getDirectedAngle(nodePoints[i] - node.point)
            if (angle < minAngle) {
                minAngle = angle
                left = nodePoints[i]
            }
            if (angle > maxAngle) {
                maxAngle = angle
                right = nodePoints[i]
            }
        }
        node.intersectConvexPoint.push([left, right])
        node.intersectConvexDegree.push(dir.angle)
    }

    order = new Array(nNode)
    for (i = 0;i < nNode;++i) {
        cnt = 0
        for (j = 0;j < nNode;++j) {
            if (i != j) {
                if (node.intersectConvexDegree[i] > node.intersectConvexDegree[j]) cnt++
            }
        }
        order[cnt] = i
    }

    for (i = 0;i < nNode;++i) {
        path = new Path()
        path.strokeColor = LANE_BORDER_COLOR
        path.strokeWidth = LANE_BORDER_WIDTH
        path.add(node.intersectConvexPoint[order[i]][1], node.intersectConvexPoint[order[(i+1) % nNode]][0])
    }
}

function _statusToColor(status) {
    switch (status) {
        case 'r':
            return 'red'
        case 'g':
            return 'green'
        default:
            return 'grey'     
    }
}
function drawStep(step) {
    stepStatus = simulation.dynamic[step]
    for (i = 0;i < stepStatus.trafficLights.length;++i) {
        trafficLight = stepStatus.trafficLights[i]
        for (j = 0;j < trafficLight.status.length;++j) {
            trafficLights[trafficLight.edge][j].strokeColor = _statusToColor(trafficLight.status[j])
        }
    }
    carLayer.removeChildren()
    carLayer.activate()
    $("#carNum").text(stepStatus.cars.length);
    for (i = 0; i < stepStatus.cars.length; ++i) {
        car = stepStatus.cars[i]
        carSymbol.place([car.point[0], 500 - car.point[1]]).rotate(car.dir / Math.PI * 180).scale(1, -1)
    }
    roadLayer.activate()
}