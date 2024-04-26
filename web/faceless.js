import { app } from '../../../scripts/app.js'
import { api } from '../../../scripts/api.js'

async function uploadFile(file) {
  try {
    const body = new FormData();
    body.append("image", file);
    const resp = await api.fetchApi("/upload/image", {
      method: "POST",
      body,
    });

    if (resp.status === 200) {
      return resp;
    } else {
      alert(resp.status + " - " + resp.statusText);
      return null;
    }
  } catch (error) {
    alert(error);
    return null;
  }
}

app.registerExtension({
  name: "Faceless.Web",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "FacelessLoadVideo") {
      const cb = nodeType.prototype.onNodeCreated;
      if (!cb) {
        return;
      }
      nodeType.prototype.onNodeCreated = function() {
        cb.apply(this, arguments)

        const videoWidget = this.widgets.find(name => name.name === "video");
        console.log("widgets", this, this.widgets, videoWidget)
        const fileInput = document.createElement("input");
        const removeCb = this.onRemoved;
        this.onRemoved = function() {
          !!removeCb && removeCb.apply(this, arguments)
          fileInput?.remove();
        }

        Object.assign(fileInput, {
          type: "file",
          accept: "video/webm,video/mp4,video/mkv,image/gif",
          style: "display: none",
          onchange: async () => {
            if (fileInput.files.length) {
              const uploadRes = await uploadFile(fileInput.files[0])
              if (uploadRes === null) {
                return;
              }

              const data = await uploadRes.json();
              let path = data.name;
              if (data.subfolder) path = data.subfolder + "/" + path;

              if (!videoWidget.options.values.includes(path)) {
                videoWidget.options.values.push(path);
              }

              videoWidget.value = path;
              // FIXME Should call the callback?
              // if (videoWidget.callback) {
              //   videoWidget.callback(path)
              // }
            }
          },
        });

        document.body.append(fileInput);
        let uploadWidget = this.addWidget("button", "choose file to upload", "image", () => {
            app.canvas.node_widget = null
            fileInput.click();
        });
        uploadWidget.options.serialize = false;
        // const widgets = this.widgets.find((w) => w.name === widgetName)
      }
    } else {
      // console.log("node data", nodeData.name, nodeData)
    }
  }
})
