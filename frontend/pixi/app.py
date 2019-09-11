from flask import Flask, request, send_from_directory, jsonify, render_template
# set the project root directory as the static folder, you can set others.
app = Flask(__name__, static_url_path='', static_folder='')

@app.route('/')
def index():
    data = {
        "roadnetFile": "replay/" + request.args.get('roadnetFile'),
        "logFile": "replay/" + request.args.get('logFile')
    }
    return render_template('index.html', data=data)

@app.route('/replay/<path:path>')
def replay(path):
    return send_from_directory('../../data/frontend/web/', path)

@app.route('/library/<path:path>')
def library(path):
    return send_from_directory('../../data/frontend/web/library/', path)

fh = {}
root = '../../data/frontend/web/'
batch = 100

def parseLine(line):
    ret = []
    carLogs, tlLogs = line.split(';')
    
    logs = []
    for carLog in carLogs.split(',')[:-1]:
        logs.append([float(x) for x in carLog.split(' ')])
    ret.append(logs)
    logs = []
    for tlLog in tlLogs.split(',')[:-1]:
        logs.append(tlLog.split(' '))
    ret.append(logs)
    return ret

@app.route('/getReplay/<path:path>')
def get_replay(path):
    frontend_id = request.args.get('id')
    if frontend_id in fh:
        f = fh[frontend_id]
    else:
        f = open(root + path)
        fh[frontend_id] = f
    logs = []
    for _ in range(batch):
        line = f.readline().strip()
        if line:
            logs.append(parseLine(line))
        else:
            f.close()
            del fh[frontend_id]
            break
    return jsonify(logs)
    

if __name__=='__main__':
    app.run(host='0.0.0.0', port=8080)