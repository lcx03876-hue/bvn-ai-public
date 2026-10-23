package net.play5d.game.obvn.ctrler.game_ctrler
{
   import flash.events.DataEvent;
   import net.play5d.game.obvn.Debugger;
   import net.play5d.game.obvn.MainGame;
   import net.play5d.game.obvn.ctrler.EffectCtrler;
   import net.play5d.game.obvn.ctrler.GameLoader;
   import net.play5d.game.obvn.ctrler.GameLogic;
   import net.play5d.game.obvn.ctrler.GameRender;
   import net.play5d.game.obvn.ctrler.SoundCtrler;
   import net.play5d.game.obvn.data.GameConfig;
   import net.play5d.game.obvn.data.GameData;
   import net.play5d.game.obvn.data.GameInterface;
   import net.play5d.game.obvn.data.GameMode;
   import net.play5d.game.obvn.data.GameRunFighterGroup;
   import net.play5d.game.obvn.data.TeamMap;
   import net.play5d.game.obvn.event.GameEvent;
   import net.play5d.game.obvn.fighter.Assister;
   import net.play5d.game.obvn.fighter.FighterAttacker;
   import net.play5d.game.obvn.fighter.FighterMain;
   import net.play5d.game.obvn.fighter.ctrler.FighterAICtrler;
   import net.play5d.game.obvn.fighter.ctrler.FighterKeyCtrler;
   import net.play5d.game.obvn.input.GameInputer;
   import net.play5d.game.obvn.interfaces.IFighterActionCtrler;
   import net.play5d.game.obvn.interfaces.IGameSprite;
   import net.play5d.game.obvn.map.MapMain;
   import net.play5d.game.obvn.model.MessionModel;
   import net.play5d.game.obvn.stage.GameStage;
   import net.play5d.game.obvn.ui.GameUI;
   import net.play5d.game.obvn.utils.KeyBoarder;
   import net.play5d.game.obvn.views.effects.BitmapFilterView;
   import net.play5d.game.obvn.vo.GameRunDataVO;
   import net.play5d.game.obvn.vo.TeamVO;
      import flash.filesystem.File;
      import flash.filesystem.FileStream;
      import flash.filesystem.FileMode;
   
   public class GameCtrler
   {
      
      private static var _i:GameCtrler;
      
      public var gameStage:GameStage;
      
      public const gameRunData:GameRunDataVO = new GameRunDataVO();
      
      public var actionEnable:Boolean = false;
      
      public var autoStartAble:Boolean = true;
      
      public var autoEndRoundAble:Boolean = true;
      
      private var _teamMap:TeamMap = new TeamMap();
      
      private var _startCtrler:GameStartCtrler;
      
      private var _fighterEventCtrler:FighterEventCtrler;
      
      private var _trainingCtrler:TrainingCtrler;
      
      private var _mainLogicCtrler:GameMainLogicCtrler;
      
      private var _endCtrler:GameEndCtrler;
      
      private var _isRenderGame:Boolean = true;
      
      private var _isPauseGame:Boolean;
      
      private var _gameRunning:Boolean;
      
      private var _renderTimeFrame:int;
      
      private var _renderAnimateGap:int = 0;
      
      private var _renderAnimateFrame:int = 0;
      
      public var fightFinished:Boolean;
      
      private var _gameStartAndPause:Boolean;
      
      public function GameCtrler()
      {
         super();
      }
      
      public static function get I() : GameCtrler
      {
         if(!_i)
         {
            _i = new GameCtrler();
         }
         return _i;
      }
      
      public function getAttacker(param1:String, param2:int) : FighterAttacker
      {
         return _fighterEventCtrler.getAttacker(param1,param2);
      }
      
      public function getAssister(param1:int) : Assister
      {
         var _loc2_:GameRunFighterGroup = gameRunData["p" + param1 + "FighterGroup"];
         if(_loc2_)
         {
            return _loc2_.fuzhu;
         }
         return null;
      }
      
      public function setRenderHit(param1:Boolean) : void
      {
         if(_mainLogicCtrler)
         {
            _mainLogicCtrler.renderHit = param1;
         }
      }
      
      public function initlize(param1:GameStage) : void
      {
         this.gameStage = param1;
         _isPauseGame = false;
         _isRenderGame = true;
         _gameRunning = true;
         _gameStartAndPause = false;
         _fighterEventCtrler = new FighterEventCtrler();
         _fighterEventCtrler.initlize();
         _renderAnimateGap = Math.ceil(GameConfig.FPS_GAME / 30) - 1;
         KeyBoarder.focus();
      }
      
      private function renderPause() : void
      {
         if(_startCtrler || _endCtrler)
         {
            if(GameInputer.back(1) || GameInputer.select("MENU",1))
            {
               if(_startCtrler)
               {
                  _startCtrler.skip();
               }
               if(_endCtrler)
               {
                  _endCtrler.skip();
               }
            }
            return;
         }
         if(GameInputer.back(1))
         {
            if(_isPauseGame)
            {
               resume(true);
            }
            else
            {
               pause(true);
            }
         }
      }
      
      public function destory() : void
      {
         GameRender.remove(render);
         GameLogic.clear();
         GameInputer.clearInput();
         if(_fighterEventCtrler)
         {
            FighterEventCtrler.destory();
            _fighterEventCtrler = null;
         }
         if(_mainLogicCtrler)
         {
            _mainLogicCtrler.destory();
            _mainLogicCtrler = null;
         }
         if(_trainingCtrler)
         {
            _trainingCtrler.destory();
            _trainingCtrler = null;
         }
         if(_startCtrler)
         {
            _startCtrler.destory();
            _startCtrler = null;
         }
         if(_endCtrler)
         {
            _endCtrler.destory();
            _endCtrler = null;
         }
         if(gameStage)
         {
            gameStage = null;
         }
         gameRunData.p1FighterGroup.destoryFighters(gameRunData.continueLoser);
         gameRunData.p2FighterGroup.destoryFighters(null);
         if(gameRunData.continueLoser == null)
         {
            gameRunData.clear();
            GameLoader.dispose();
         }
         _gameRunning = false;
      }
      
      public function getEnemyTeam(param1:IGameSprite) : TeamVO
      {
         if(param1.team)
         {
            return _teamMap.getTeam(param1.team.id == 1 ? 2 : 1);
         }
         return null;
      }
      
      public function addGameSprite(param1:int, param2:IGameSprite, param3:int = -1) : void
      {
         if(param3 != -1)
         {
            gameStage.addGameSpriteAt(param2,param3);
         }
         else
         {
            gameStage.addGameSprite(param2);
         }
         var _loc4_:TeamVO = _teamMap.getTeam(param1);
         if(_loc4_)
         {
            param2.team = _loc4_;
            _loc4_.addChild(param2);
            if(param2 is FighterMain)
            {
               (param2 as FighterMain).targetTeams = _teamMap.getOtherTeams(param1);
            }
         }
         else if(!(param2 is BitmapFilterView))
         {
            Debugger.log("GameCtrl.addGameSprite :: team 为空！");
         }
      }
      
      public function removeGameSprite(param1:IGameSprite, param2:Boolean = false) : void
      {
         gameStage.removeGameSprite(param1);
         var _loc3_:TeamVO = param1.team;
         if(_loc3_)
         {
            _loc3_.removeChild(param1);
         }
         param1.destory(param2);
      }
      
      public function startGame() : void
      {
         if(!autoStartAble)
         {
            return;
         }
         fightFinished = false;
         doStartGame();
      }
      
      public function doStartGame() : void
      {
         _isPauseGame = false;
         GameInputer.enabled = true;
         gameRunData.reset();
         initTeam();
         buildGame();
         GameEvent.dispatchEvent("GAME_START");
         GameRender.add(render);
      }
      
      private function buildGame() : void
      {
         var _loc2_:FighterMain = gameRunData.p1FighterGroup.currentFighter;
         var _loc1_:FighterMain = gameRunData.p2FighterGroup.currentFighter;
         if(GameMode.currentMode == 40)
         {
            _trainingCtrler = new TrainingCtrler();
            _trainingCtrler.initlize([_loc2_,_loc1_]);
            gameRunData.gameTimeMax = -1;
         }
         var _loc3_:MapMain = gameRunData.map;
         if(!_loc2_ || !_loc1_ || !_loc3_)
         {
            throw new Error("Game creation failed!");
         }
         addFighter(_loc2_,1);
         addFighter(_loc1_,2);
         _loc3_.initlize();
         gameStage.initFight(gameRunData.p1FighterGroup,gameRunData.p2FighterGroup,_loc3_);
         GameLogic.initGameLogic(_loc3_,gameStage.camera);
         _mainLogicCtrler = new GameMainLogicCtrler();
         _mainLogicCtrler.initlize(gameStage,_teamMap,_loc3_);
         if(GameMode.currentMode == 40)
         {
            actionEnable = true;
            GameUI.I.fadIn();
            SoundCtrler.I.playFightBGM("map");
         }
         else
         {
            _startCtrler = new GameStartCtrler(gameStage);
            actionEnable = false;
            _startCtrler.start1v1(_loc2_,_loc1_);
         }
         GameInterface.instance.afterBuildGame();
      }
      
      private function addFighter(param1:FighterMain, param2:int) : void
      {
         var _loc3_:IFighterActionCtrler = null;
         if(!param1)
         {
            return;
         }
         switch(param2 - 1)
         {
            case 0:
               if(GameMode.isWatch())
               {
                  _loc3_ = new FighterAICtrler();
                  (_loc3_ as FighterAICtrler).AILevel = MessionModel.I.AI_LEVEL;
                  (_loc3_ as FighterAICtrler).fighter = param1;
                  break;
               }
               _loc3_ = new FighterKeyCtrler();
               (_loc3_ as FighterKeyCtrler).inputType = "P1";
               (_loc3_ as FighterKeyCtrler).classicMode = GameData.I.config.keyInputMode == 1;
               break;
            case 1:
               if(GameMode.isVsCPU(false) || GameMode.isAcrade())
               {
                  _loc3_ = new FighterAICtrler();
                  (_loc3_ as FighterAICtrler).AILevel = MessionModel.I.AI_LEVEL;
                  (_loc3_ as FighterAICtrler).fighter = param1;
                  break;
               }
               _loc3_ = new FighterKeyCtrler();
               (_loc3_ as FighterKeyCtrler).inputType = "P2";
               (_loc3_ as FighterKeyCtrler).classicMode = GameData.I.config.keyInputMode == 1;
         }
         param1.initlize();
         param1.setActionCtrl(_loc3_);
         addGameSprite(param2,param1);
      }
      
      private function removeFighter(param1:FighterMain) : void
      {
         if(!param1)
         {
            return;
         }
         removeGameSprite(param1);
      }
      
      private function buildNextRound(param1:Boolean) : void
      {
         doBuildNextRound(param1);
      }
      
      private function doBuildNextRound(param1:Boolean) : void
      {
         var _loc2_:int = 0;
         gameStage.resetFight(gameRunData.p1FighterGroup,gameRunData.p2FighterGroup);
         _startCtrler = new GameStartCtrler(gameStage);
         if(param1)
         {
            if(gameRunData.lastWinner)
            {
               gameRunData.lastWinner.hp = gameRunData.lastWinnerHp;
            }
            _loc2_ = -1;
            if(gameRunData.lastWinnerTeam)
            {
               _loc2_ = gameRunData.lastWinnerTeam.id == 1 ? 2 : 1;
            }
            _startCtrler.start1v1(gameRunData.p1FighterGroup.currentFighter,gameRunData.p2FighterGroup.currentFighter,_loc2_);
         }
         else
         {
            _startCtrler.startNextRound();
         }
         gameRunData.isDrawGame = false;
         GameEvent.dispatchEvent("ROUND_START");
      }
      
      public function fightFinish() : void
      {
         fightFinished = true;
         if(GameMode.isAcrade())
         {
            if(gameRunData.lastWinnerTeam.id == 1)
            {
               if(MessionModel.I.missionAllComplete())
               {
                  trace("关卡全部通过！");
                  MainGame.I.goCongratulations();
               }
               else
               {
                  trace("下一关卡");
                  GameData.I.winnerId = gameRunData.p1FighterGroup.currentFighter.data.id;
                  MainGame.I.goWinner();
               }
            }
            else
            {
               trace("前往继续场景");
               gameRunData.continueLoser = gameRunData.p1FighterGroup.currentFighter;
               MainGame.I.goContinue();
            }
         }
         if(GameMode.isVsCPU() || GameMode.isVsPeople())
         {
            trace("返回选人菜单");
            GameEvent.dispatchEvent("GAME_END");
            MainGame.I.goSelect();
         }
      }
      
      private function initTeam() : void
      {
         _teamMap.clear();
         var _loc1_:Array = GameMode.getTeams();
         for each(var _loc2_:Object in _loc1_)
         {
            _teamMap.add(new TeamVO(_loc2_.id,_loc2_.name));
         }
      }
      
      public function pause(param1:Boolean = false) : void
      {
         if(!_gameRunning)
         {
            return;
         }
         if(param1 && !_isPauseGame)
         {
            if(_startCtrler || _endCtrler)
            {
               _gameStartAndPause = true;
               return;
            }
            GameEvent.dispatchEvent("PAUSE_GAME");
            _isPauseGame = true;
            GameUI.I.getUI().pause();
            MainGame.I.stage.dispatchEvent(new DataEvent("5d_message",false,false,JSON.stringify(["game_pause"])));
         }
         _isRenderGame = false;
      }
      
      public function resume(param1:Boolean = false) : void
      {
         if(!_gameRunning)
         {
            return;
         }
         _gameStartAndPause = false;
         if(param1 && _isPauseGame)
         {
            GameEvent.dispatchEvent("RESUME_GAME");
            _isPauseGame = false;
            GameUI.I.getUI().resume();
            MainGame.I.stage.dispatchEvent(new DataEvent("5d_message",false,false,JSON.stringify(["game_resume"])));
         }
         KeyBoarder.focus();
         _isRenderGame = true;
      }
      
      public function gameEnd(param1:FighterMain, param2:FighterMain) : void
      {
         if(!autoEndRoundAble)
         {
            return;
         }
         if(_endCtrler)
         {
            return;
         }
         doGameEnd(param1,param2);
      }
      
      public function doGameEnd(param1:FighterMain, param2:FighterMain) : void
      {
         gameRunData.lastWinnerTeam = param1.team;
         gameRunData.lastWinner = param1;
         gameRunData.lastLoserData = param2.data;
         gameRunData.lastLoserQi = param2.qi;
         switch(param1.team.id - 1)
         {
            case 0:
               gameRunData.p1Wins++;
               if(param2.hp <= 0 && GameMode.isAcrade())
               {
                  GameLogic.addScoreByKO();
               }
               break;
            case 1:
               gameRunData.p2Wins++;
         }
         _endCtrler = new GameEndCtrler();
         _endCtrler.initlize(param1,param2);
         actionEnable = false;
         GameEvent.dispatchEvent("ROUND_END");
      }
      
      private function render() : void
      {
         bvnExportState();
         var _loc2_:Boolean = false;
         var _loc1_:Boolean = false;
         renderPause();
         if(_isPauseGame)
         {
            return;
         }
         EffectCtrler.I.render();
         gameStage.render();
         if(!_isRenderGame)
         {
            return;
         }
         checkRenderAnimate();
         if(_mainLogicCtrler)
         {
            _mainLogicCtrler.render();
         }
         if(_startCtrler)
         {
            actionEnable = false;
            _loc2_ = _startCtrler.render();
            if(_loc2_)
            {
               _startCtrler.destory();
               _startCtrler = null;
               actionEnable = true;
               gameRunData.setAllowLoseHP(true);
               if(_gameStartAndPause)
               {
                  pause(true);
                  _gameStartAndPause = false;
               }
            }
         }
         if(_endCtrler)
         {
            _loc1_ = _endCtrler.render();
            if(_loc1_)
            {
               _endCtrler.destory();
               _endCtrler = null;
               runNext();
            }
         }
         if(_trainingCtrler)
         {
            _trainingCtrler.render();
         }
      }
      
      private function checkRenderAnimate() : void
      {
         if(_renderAnimateGap > 0)
         {
            if(_renderAnimateFrame++ >= _renderAnimateGap)
            {
               _renderAnimateFrame = 0;
               renderAnimate();
            }
         }
         else
         {
            renderAnimate();
         }
      }
      
      private function renderAnimate() : void
      {
         if(_mainLogicCtrler)
         {
            _mainLogicCtrler.renderAnimate();
         }
         if(actionEnable && !_startCtrler && !_endCtrler)
         {
            renderGameTime();
         }
      }
      
      private function renderGameTime() : void
      {
         if(gameRunData.gameTimeMax != -1)
         {
            if(++_renderTimeFrame > 30)
            {
               _renderTimeFrame = 0;
               gameRunData.gameTime--;
               if(gameRunData.gameTime <= 0)
               {
                  timeover();
               }
            }
         }
      }
      
      private function timeover() : void
      {
         trace("时间结束！");
         actionEnable = false;
         var _loc1_:FighterMain = gameRunData.p1FighterGroup.currentFighter;
         var _loc2_:FighterMain = gameRunData.p2FighterGroup.currentFighter;
         gameRunData.isTimerOver = true;
         if(_loc1_.hp == _loc2_.hp)
         {
            drawGame();
            return;
         }
         if(_loc1_.hp > _loc2_.hp)
         {
            gameEnd(_loc1_,_loc2_);
         }
         else
         {
            gameEnd(_loc2_,_loc1_);
         }
      }
      
      public function drawGame() : void
      {
         if(_endCtrler)
         {
            return;
         }
         gameRunData.lastWinnerTeam = null;
         gameRunData.lastWinner = null;
         gameRunData.isDrawGame = true;
         _endCtrler = new GameEndCtrler();
         _endCtrler.drawGame();
         actionEnable = false;
      }
      
      private function runNext() : void
      {
         gameRunData.nextRound();
         if(GameMode.isTeamMode())
         {
            if(startNextTeamFight())
            {
               buildNextRound(true);
               gameRunData.lastWinner = null;
               return;
            }
         }
         if(GameMode.isSingleMode())
         {
            if(gameRunData.p1Wins < 2 && gameRunData.p2Wins < 2)
            {
               buildNextRound(false);
               gameRunData.lastWinner = null;
               return;
            }
         }
         fightFinish();
      }
      
      private function startNextTeamFight() : Boolean
      {
         var _loc1_:FighterMain = null;
         var _loc2_:FighterMain = null;
         if(gameRunData.isDrawGame)
         {
            _loc1_ = gameRunData.p1FighterGroup.getNextFighter();
            _loc2_ = gameRunData.p2FighterGroup.getNextFighter();
            if(!_loc1_ && !_loc2_)
            {
               return true;
            }
            if(_loc1_ && !_loc2_)
            {
               gameRunData.lastWinnerTeam = gameRunData.p1FighterGroup.currentFighter.team;
               return false;
            }
            if(!_loc1_ && _loc2_)
            {
               gameRunData.lastWinnerTeam = gameRunData.p2FighterGroup.currentFighter.team;
               return false;
            }
            nextFighter(gameRunData.p1FighterGroup);
            nextFighter(gameRunData.p2FighterGroup);
            return true;
         }
         switch(gameRunData.lastWinnerTeam.id - 1)
         {
            case 0:
               return nextFighter(gameRunData.p2FighterGroup);
            case 1:
               return nextFighter(gameRunData.p1FighterGroup);
            default:
               gameRunData.lastWinnerTeam = null;
               return true;
         }
      }
      
      private function nextFighter(param1:GameRunFighterGroup) : Boolean
      {
         if(!param1)
         {
            return false;
         }
         var _loc2_:TeamVO = param1.currentFighter.team;
         var _loc3_:FighterMain = param1.getNextFighter();
         if(!_loc3_)
         {
            return false;
         }
         if(gameRunData.lastLoserData)
         {
            if(gameRunData.lastLoserData.comicType == _loc3_.data.comicType)
            {
               _loc3_.qi = gameRunData.lastLoserQi + 100;
            }
            else
            {
               _loc3_.qi = gameRunData.lastLoserQi;
            }
            if(_loc3_.qi > 300)
            {
               _loc3_.qi = 300;
            }
         }
         removeFighter(param1.currentFighter);
         param1.removeCurrentFighter();
         param1.currentFighter = _loc3_;
         addFighter(param1.currentFighter,_loc2_.id);
         return true;
      }
      
      public function slow(param1:Number) : void
      {
         var _loc2_:Number = 30 / param1;
         setAnimateFPS(_loc2_);
         _mainLogicCtrler.setSpeedPlus(GameConfig.SPEED_PLUS_DEFAULT / param1);
         gameStage.camera.tweenSpd = 2.5 * param1;
      }
      
      public function slowResume() : void
      {
         setAnimateFPS(30);
         _mainLogicCtrler.setSpeedPlus(GameConfig.SPEED_PLUS_DEFAULT);
         gameStage.camera.tweenSpd = 2.5;
      }
      
      private function setAnimateFPS(param1:Number) : void
      {
         _renderAnimateGap = Math.ceil(GameConfig.FPS_GAME / param1) - 1;
         _renderAnimateFrame = 0;
      }
      
      public function get isRenderGame() : Boolean
      {
         return _isRenderGame;
      }
      
      public function get isPauseGame() : Boolean
      {
         return _isPauseGame;
      }
   
      // ==================== 状态导出桥接（注入代码，非原版）====================
      // 每帧把双方状态写成一个 JSON 文件，Python 侧轮询读取。
      // 这样完全绕开内存扫描：不用担心 AS3 的 GC 换地址、
      // 也不用在一堆同结构的克隆对象里猜哪个是活的。
      private static var _bvnFrame:int = 0;
      // ★ 开局检测（用户要求）★
      //   `_bvnAeFrame` = actionEnable 最近一次 false->true 的帧号；
      //   `_bvnPrevAe`  = 上一帧的 actionEnable，用来抓跳变。
      //   导出 `aeFrames`（回合年龄）后，控制器就能在回合刚开的瞬间
      //   强制"松开所有键 + 重按"，避免「开局站着不动」。
      private static var _bvnAeFrame:int = 0;
      private static var _bvnPrevAe:Boolean = false;
      private static var _bvnFile:File = null;
      private static var _bvnStream:FileStream = null;

      private function bvnDumpFighter(param1:Object) : String
      {
         if(param1 == null)
         {
            return "null";
         }
         var out:String = "{";
         try
         {
            out += "\"hp\":" + param1.hp;
            out += ",\"hpMax\":" + param1.hpMax;
            out += ",\"qi\":" + param1.qi;
            out += ",\"energy\":" + param1.energy;
            out += ",\"energyMax\":" + param1.energyMax;
            out += ",\"x\":" + param1.x;
            out += ",\"y\":" + param1.y;
            out += ",\"actionState\":" + param1.actionState;
            // 朝向字段叫 direct（BaseGameSprite._direct，有 public getter），
            // **不叫 facing** —— 之前按 facing 找是找不到的
            out += ",\"direct\":" + param1.direct;
            out += ",\"heavy\":" + param1.heavy;
            out += ",\"defense\":" + param1.defense;
            out += ",\"isInAir\":" + (param1.isInAir ? "true" : "false");
            out += ",\"isAlive\":" + (param1.isAlive ? "true" : "false");
            out += ",\"speed\":" + param1.speed;
         }
         catch(bvnErr:Error)
         {
            out += "\"err\":\"" + bvnErr.message + "\"";
         }
         return out + "}";
      }

      private function bvnCharId(param1:Object) : String
      {
         // ★ 角色 id（如 "ichigo" / "naruto" / "pain"）★
         // 为什么要它：我一直靠**看画面**猜"哪一方是一护"，
         // 但一护和鸣人都是橙色头发 -> 认错过。读数据最可靠。
         // 依据：LoadingStage.as 里就是用 `param1.data.id` 取角色 id 的。
         if(param1 == null)
         {
            return "null";
         }
         try
         {
            var dt:Object = param1.data;
            if(dt == null)
            {
               return "null";
            }
            return "\"" + dt.id + "\"";
         }
         catch(bvnErr4:Error)
         {
            return "\"err\"";
         }
      }

      private function bvnHasWankai(param1:Object) : String
      {
         // FighterMain.hasWankai() -> MC 里有没有 "万解" 帧
         if(param1 == null)
         {
            return "false";
         }
         try
         {
            return param1.hasWankai() ? "true" : "false";
         }
         catch(bvnErr6:Error)
         {
            return "false";
         }
      }

      private function bvnQuoted(param1:String) : String
      {
         if(param1 == null || param1 == "")
         {
            return "null";
         }
         return "\"" + param1 + "\"";
      }

      private function bvnFzId(param1:Object) : String
      {
         // 辅助 id：GameRunFighterGroup.fuzhu
         if(param1 == null)
         {
            return "null";
         }
         try
         {
            // ⚠️ `GameRunFighterGroup.fuzhu` 是 **Assister 对象**，不是字符串。
            //    实测直接取它会得到 "[object Assister]"。
            //    `Assister` 有 `public var data:FighterVO`，所以 id 在 `data.id`。
            var bvnDt:Object = param1.data;
            if(bvnDt == null)
            {
               return "null";
            }
            return bvnQuoted(bvnDt.id);
         }
         catch(bvnErr5:Error)
         {
            return "null";
         }
      }

      private function bvnIsAI(param1:Object) : String
      {
         // ★ 确定性判据：沿 getCtrler().getMcCtrl().getActionCtrler() 拿到
         //   真正的动作控制器，再看它是不是 FighterAICtrler ★
         //
         // 踩过的坑：`fighter.getCtrler()` 返回的是 **FighterCtrler**（不是动作控制器），
         // 直接写 `getCtrler() is FighterAICtrler` **永远为 false**。
         // 动作控制器存在 FighterCtrler._fighterMcCtrler._actionCtrler 里，
         // 而 FighterMcCtrler 原本**没有 public getter** ——
         // 所以 build_bridge 还得给 FighterMcCtrler 补一个 getActionCtrler()。
         if(param1 == null)
         {
            return "null";
         }
         try
         {
            var c:Object = param1.getCtrler();
            if(c == null)
            {
               return "null";
            }
            var mc:Object = c.getMcCtrl();
            if(mc == null)
            {
               return "null";
            }
            var ac:Object = mc.getActionCtrler();
            if(ac == null)
            {
               return "null";
            }
            return (ac is FighterAICtrler) ? "true" : "false";
         }
         catch(bvnErr3:Error)
         {
            return "\"err:" + bvnErr3.message + "\"";
         }
      }

      private function bvnExportState() : void
      {
         _bvnFrame++;
         // ★ 开局检测：抓 actionEnable 的 false->true 跳变 ★
         if(actionEnable && !_bvnPrevAe)
         {
            _bvnAeFrame = _bvnFrame;
         }
         _bvnPrevAe = actionEnable;
         if(_bvnFrame % 2 != 0)
         {
            return;                      // 约 30Hz，够 AI 用，省 IO
         }
         try
         {
            var p1:Object = null;
            var p2:Object = null;
            if(gameRunData != null)
            {
               if(gameRunData.p1FighterGroup != null)
               {
                  p1 = gameRunData.p1FighterGroup.currentFighter;
               }
               if(gameRunData.p2FighterGroup != null)
               {
                  p2 = gameRunData.p2FighterGroup.currentFighter;
               }
            }
            var s:String = "{\"frame\":" + _bvnFrame
               + ",\"actionEnable\":" + (actionEnable ? "true" : "false")
            // ★ 开局检测字段（用户要求）★
            //   aeFrame  = 最近一次可操作的起始游戏帧
            //   aeFrames = 距那次过了几帧（-1 = 当前不可操作）
            //   roundNo  = 第几小局（三局两胜，先到 2 胜）
            + ",\"aeFrame\":" + _bvnAeFrame
            + ",\"aeFrames\":" + (actionEnable ? (_bvnFrame - _bvnAeFrame) : -1)
            + ",\"roundNo\":" + (gameRunData != null ? (gameRunData.p1Wins + gameRunData.p2Wins + 1) : -1)
               // 运行时 AI 等级 —— 用来确认 set_difficulty 到底生效没有。
               // config.AI_level = FighterTester 里硬编码那个；
               // MessionModel.AI_LEVEL = 真正传进 FighterAICtrler 那个（选角时赋值）。
               + ",\"aiLevel\":" + GameData.I.config.AI_level
               + ",\"aiModel\":" + MessionModel.I.AI_LEVEL
               + ",\"mode\":" + GameMode.currentMode
               // ★ 三局两胜的官方比分 ★
               //   GameCtrler.runNext(): if(p1Wins < 2 && p2Wins < 2) 才 buildNextRound
               //   => 先到 2 胜整场才结束。判断"整场是否结束"必须看这两个，别靠猜。
               + ",\"p1Wins\":" + gameRunData.p1Wins
               + ",\"p2Wins\":" + gameRunData.p2Wins
               // ★ 两方各自的控制器类型（是不是 AI）—— 模式检测的硬证据 ★
               // ★ 能不能变身（万解）★ —— 用来验证"变身"这条动作有没有生效
               + ",\"p1Wankai\":" + bvnHasWankai(p1)
               + ",\"p2Wankai\":" + bvnHasWankai(p2)
               + ",\"p1IsAI\":" + bvnIsAI(p1)
               + ",\"p2IsAI\":" + bvnIsAI(p2)
               // ★ 角色 id —— 别再靠看画面猜了（看画面我把佩恩认成了一护）★
               + ",\"p1Id\":" + bvnCharId(p1)
               + ",\"p2Id\":" + bvnCharId(p2)
               // ★ 辅助 id（2026-10-15）★
               //   为什么要：实测面板曾把辅助值写进角色位
               //   （玩家1 角色id = kon），而桥接原来**不导出辅助 id**，
               //   所以这种错误没被发现就直接开局了。
               //   来源 `GameRunFighterGroup.fuzhu`（运行时权威值）。
               + ",\"p1Fz\":" + bvnFzId(gameRunData.p1FighterGroup)
               + ",\"p2Fz\":" + bvnFzId(gameRunData.p2FighterGroup)
               // 选角 VO 的原始值（面板写进去的那份，用于交叉核对）
               + ",\"p1Sel\":" + bvnQuoted(GameData.I.p1Select
                  ? GameData.I.p1Select.fuzhu : "")
               + ",\"p2Sel\":" + bvnQuoted(GameData.I.p2Select
                  ? GameData.I.p2Select.fuzhu : "")
               + ",\"p1\":" + bvnDumpFighter(p1)
               + ",\"p2\":" + bvnDumpFighter(p2)
               + "}";
            if(_bvnFile == null)
            {
               _bvnFile = new File("C:\\deepseek\\bvn-ai\\runs\\bvn_state.json");
            }
            if(_bvnStream == null)
            {
               _bvnStream = new FileStream();
            }
            _bvnStream.open(_bvnFile, FileMode.WRITE);
            _bvnStream.writeUTFBytes(s);
            _bvnStream.close();
         }
         catch(bvnErr2:Error)
         {
            // 导出失败绝不能影响游戏本体，静默吞掉
         }
      }
      // ================== 状态导出桥接结束 ==================

}
}
