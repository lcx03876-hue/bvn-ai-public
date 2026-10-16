      // Match mode used by the "start test" button.
      //   22 = single player vs CPU (P2 is driven by FighterAICtrler) -- default
      //   40 = practice mode (the stock behaviour)
      // Toggle with F2; see the listener in buildTestUI().
      //
      // Why 22 is the default: the F2 "switch mode" path proved unreliable in
      // practice (the stage often does not receive keyboard events), whereas
      // the "start test" button is dependable. So the commonly used mode is the
      // default and F2 is only a fallback.
      private var _testMode:int = 22;

      // Panel self-report export.
      //
      // The panel has silently produced wrong values before: player 1's
      // character id was written into the assist field because the field
      // detection had shifted by one row. To catch that class of error the
      // panel's *actual* text field contents are written to disk every 4
      // frames, and the Python side reads them back after typing -- no OCR,
      // no guessing.
      private var _bvnPanelFile:File;
      private var _bvnPanelStream:FileStream;
      private var _bvnPanelFrame:int = 0;
