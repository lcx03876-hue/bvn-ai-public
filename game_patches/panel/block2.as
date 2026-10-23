      // ★ 「开始测试」要用的对局模式 ★
      //   22 = 电脑对战-单人（P2 是 FighterAICtrler）—— **默认**
      //   40 = 练习模式（原版行为）
      //   用 F2 切换，见 buildTestUI() 里的监听。
      //
      //   为什么默认改成 22：F2 那条"切模式"的路实测不可靠
      //   （stage 收不到键盘事件），而「开始测试」按钮很可靠。
      //   所以把常用模式设成默认值，F2 只当备用。
      private var _testMode:int = 22;

      // ★ 面板自省导出（2026-10-15）★
      //   用户要求「开局也应该写个检查」。
      //   实测面板出过错：玩家1 角色id 被写成 kon（字段错位）。
      //   这里把**面板输入框的真实内容**每 4 帧写到
      //   C:\deepseek\bvn-ai\runs\panel_state.json，
      //   Python 写完字段后读回核对 —— 不需要 OCR，也不靠猜。
      private var _bvnPanelFile:File;
      private var _bvnPanelStream:FileStream;
      private var _bvnPanelFrame:int = 0;
