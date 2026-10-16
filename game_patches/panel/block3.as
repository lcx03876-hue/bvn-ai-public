         // Start the panel self-report export (writes every 4th frame).
         addEventListener(Event.ENTER_FRAME, function(param1:Event):void
         {
            bvnExportPanel();
         });
         // Let keyboard events reach the stage.
         //
         // After typing into a TextField the focus stays there, and a focused
         // TextField *consumes* key events, so stage listeners never fire
         // (observed: F1 appeared to do nothing). Clearing the focus as soon as
         // the field loses it lets keys bubble to the stage again.
         stage.addEventListener(FocusEvent.FOCUS_OUT, function(param1:FocusEvent):void
         {
            if(stage)
            {
               stage.focus = null;
            }
         });
         // F2: toggle between practice (40) and vs-CPU (22).
         //
         // A hotkey that called loadGame() directly never fired reliably (the
         // stage's keyboard listener is affected by panel focus and timing), so
         // this only flips the mode and lets the dependable "start test" button
         // do the rest.
         stage.addEventListener(KeyboardEvent.KEY_UP, function(param1:KeyboardEvent):void
         {
            if(param1.keyCode != 113)          // F2
            {
               return;
            }
            _testMode = _testMode == 40 ? 22 : 40;
            _debugText.text = "[F2] test mode = " + _testMode
               + (_testMode == 22 ? " (vs CPU, single)" : " (practice)");
         });
         // ============ INJECTED: hotkey to enter vs-CPU directly (not original) ============
         // Purpose: bypass the character-select screen ignoring the panel fields.
         // `testClickHandler` already proved that "set GameData, then call
         // loadGame()" starts a match, so this copies that approach and only
         // changes the mode to 22 (vs CPU, single player -> P2 is an
         // FighterAICtrler).
         //
         // Bound to F1: on the character-select screen, pressing F1 starts a
         // vs-CPU match with the characters currently in the panel.
         // KEY_UP is used because games usually handle actions on KEY_DOWN.
         // The whole body is wrapped in try/catch and reports into the panel's
         // error text field, so a failure to fire is immediately visible.
         stage.addEventListener(KeyboardEvent.KEY_UP, function(param1:KeyboardEvent):void
         {
            if(param1.keyCode != 112)          // F1 only
            {
               return;
            }
            try
            {
               // Belt and braces: drop focus from any text field first.
               stage.focus = null;
               _debugText.text = "[F1] entering hotkey handler ...";
               GameMode.currentMode = 22;      // vs CPU - single
               TrainingCtrler.RECOVER_HP = _autoReceiveHp.text != "0";
               GameData.I.p1Select = new SelectVO();
               GameData.I.p2Select = new SelectVO();
               GameData.I.p1Select.fighter1 = _p1InputId.text;
               GameData.I.p2Select.fighter1 = _p2InputId.text;
               GameData.I.p1Select.fuzhu = _p1FzInputId.text;
               GameData.I.p2Select.fuzhu = _p2FzInputId.text;
               GameData.I.selectMap = _mapInputId.text;
               _debugText.text = "[F1] p1=" + _p1InputId.text
                  + " p2=" + _p2InputId.text + " mode=22 -> loadGame()";
               loadGame();
            }
            catch(bvnHotErr:Error)
            {
               _debugText.text = "[F1][ERR] " + bvnHotErr.message;
            }
         });
         // ================== end of hotkey injection ==================
