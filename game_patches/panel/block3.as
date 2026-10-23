         // ★ 启动面板自省导出（每 4 帧写一次）★
         addEventListener(Event.ENTER_FRAME, function(param1:Event):void
         {
            bvnExportPanel();
         });
         // ★ 让按键能正常到达 stage ★
         //   原因：刚在输入框里打完字时，焦点在 TextField 上，
         //   TextField 会**消费按键事件**，stage 的监听器收不到（实测 F1 无反应）。
         //   这里"一旦离开输入框就清空焦点"，按键就会正常冒泡到 stage。
         stage.addEventListener(FocusEvent.FOCUS_OUT, function(param1:FocusEvent):void
         {
            if(stage)
            {
               stage.focus = null;
            }
         });
         // ★ F2：在「练习模式(40)」和「电脑对战(22)」之间切换 ★
         //   为什么不用热键直接 loadGame()：那条路实测一直没触发（stage 键盘监听
         //   受面板焦点/时机影响）。改成"切模式 + 点已可靠的「开始测试」按钮"。
         stage.addEventListener(KeyboardEvent.KEY_UP, function(param1:KeyboardEvent):void
         {
            if(param1.keyCode != 113)          // F2
            {
               return;
            }
            _testMode = _testMode == 40 ? 22 : 40;
            _debugText.text = "[F2] 测试模式 = " + _testMode
               + (_testMode == 22 ? " (电脑对战-单人)" : " (练习模式)");
         });
         // ==================== 注入：热键直接开电脑对战（非原版）====================
         // 目的：绕开"选角界面不吃面板字段"的问题（见 HANDOFF X 节）。
         // `testClickHandler` 已经证明"设好 GameData 再 loadGame() 就能开局"，
         // 这里照抄它，只把模式改成 22（电脑对战-单人 -> P2 是 FighterAICtrler）。
         //
         // 绑定 F1：进选角界面后按 F1，直接以面板里的角色进电脑对战。
         // 用 KEY_UP：游戏一般把动作处理放在 KEY_DOWN，KEY_UP 更容易收到。
         // 整个处理体包 try/catch，并把结果/错误**写进面板的「错误信息提示」框**
         // （`_debugText`），这样"没触发"也能立刻看出原因。
         stage.addEventListener(KeyboardEvent.KEY_UP, function(param1:KeyboardEvent):void
         {
            if(param1.keyCode != 112)          // 只看 F1
            {
               return;
            }
            try
            {
               // 双保险：先把焦点从任何输入框拿掉，再继续
               stage.focus = null;
               _debugText.text = "[F1] 进入热键处理 ...";
               GameMode.currentMode = 22;      // 电脑对战 - 单人
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
         // ================== 热键注入结束 ==================
