      private function bvnExportPanel() : void
      {
         // 把面板输入框的当前内容写成 JSON，供 Python 开局前核对。
         // 写失败绝不影响游戏（静默吞掉）。
         if(++_bvnPanelFrame % 4 != 0)
         {
            return;
         }
         try
         {
            var s:String = "{"
               + "\"p1Id\":" + bvnQ(_p1InputId) + ","
               + "\"p1Fz\":" + bvnQ(_p1FzInputId) + ","
               + "\"p2Id\":" + bvnQ(_p2InputId) + ","
               + "\"p2Fz\":" + bvnQ(_p2FzInputId) + ","
               + "\"mapId\":" + bvnQ(_mapInputId) + ","
               + "\"fps\":" + bvnQ(_fpsInput) + ","
               + "\"recvHp\":" + bvnQ(_autoReceiveHp) + ","
               + "\"testMode\":" + _testMode
               + "}";
            if(_bvnPanelFile == null)
            {
               _bvnPanelFile = new File("C:\\deepseek\\bvn-ai\\runs\\panel_state.json");
            }
            if(_bvnPanelStream == null)
            {
               _bvnPanelStream = new FileStream();
            }
            _bvnPanelStream.open(_bvnPanelFile, FileMode.WRITE);
            _bvnPanelStream.writeUTFBytes(s);
            _bvnPanelStream.close();
         }
         catch(bvnPanelErr:Error)
         {
         }
      }

      private function bvnQ(param1:Object) : String
      {
         // 把 TextField 的 text 变成 JSON 字符串（带引号 + 最简转义）
         if(param1 == null)
         {
            return "null";
         }
         var v:String = param1.text;
         if(v == null)
         {
            v = "";
         }
         v = v.split("\\").join("\\\\");
         v = v.split("\"").join("\\\"");
         return "\"" + v + "\"";
      }

