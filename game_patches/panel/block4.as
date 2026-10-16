      private function bvnExportPanel() : void
      {
         // Write the panel's current field contents as JSON so the Python side
         // can verify them before starting a match. Failures are swallowed --
         // this must never affect the game.
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
               // The path is substituted by patch_debug_panel.py from
               // bvn_paths.panel_state_path_as3().
               _bvnPanelFile = new File("__PANEL_STATE_PATH__");
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
         // Turn a TextField's text into a JSON string (quoted, minimal escapes).
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
