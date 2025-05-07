import pygame
import random
import time

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BLOCK_SIZE = 30
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
GAME_AREA_LEFT = (SCREEN_WIDTH - BOARD_WIDTH * BLOCK_SIZE) // 2
GAME_AREA_TOP = (SCREEN_HEIGHT - BOARD_HEIGHT * BLOCK_SIZE) // 2

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)

# Tetromino shapes and colors
SHAPES = [
    [[1, 1, 1, 1]],  # I
    [[1, 1], [1, 1]],  # O
    [[1, 1, 1], [0, 1, 0]],  # T
    [[1, 1, 1], [1, 0, 0]],  # J
    [[1, 1, 1], [0, 0, 1]],  # L
    [[0, 1, 1], [1, 1, 0]],  # S
    [[1, 1, 0], [0, 1, 1]]   # Z
]

COLORS = [CYAN, YELLOW, MAGENTA, BLUE, ORANGE, GREEN, RED]

class Tetris:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Tetris")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)
        
        self.reset_game()
        
    def reset_game(self):
        # Initialize game board (0 means empty)
        self.board = [[0 for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
        self.score = 0
        self.level = 1
        self.game_over = False
        self.new_piece()
        self.last_drop_time = time.time()
        
    def new_piece(self):
        """Create a new falling tetromino"""
        self.piece_index = random.randint(0, len(SHAPES) - 1)
        self.current_piece = SHAPES[self.piece_index]
        self.piece_color = COLORS[self.piece_index]
        self.piece_x = BOARD_WIDTH // 2 - len(self.current_piece[0]) // 2
        self.piece_y = 0
        
        # Check if the new piece can be placed at the starting position
        if not self.is_valid_position():
            self.game_over = True
            
    def rotate_piece(self):
        """Rotate the current piece clockwise"""
        rotated = list(zip(*reversed(self.current_piece)))
        rotated = [list(row) for row in rotated]
        
        old_piece = self.current_piece
        self.current_piece = rotated
        
        # If the rotation causes a collision, revert back
        if not self.is_valid_position():
            self.current_piece = old_piece
            
    def is_valid_position(self):
        """Check if the current piece is in a valid position"""
        for y, row in enumerate(self.current_piece):
            for x, cell in enumerate(row):
                if cell:
                    # Check if outside the board boundaries
                    if (self.piece_x + x < 0 or self.piece_x + x >= BOARD_WIDTH or
                            self.piece_y + y >= BOARD_HEIGHT):
                        return False
                    # Check if overlapping with existing blocks on the board
                    if self.piece_y + y >= 0 and self.board[self.piece_y + y][self.piece_x + x]:
                        return False
        return True
        
    def place_piece(self):
        """Place the falling piece onto the board"""
        for y, row in enumerate(self.current_piece):
            for x, cell in enumerate(row):
                if cell and 0 <= self.piece_y + y < BOARD_HEIGHT:
                    self.board[self.piece_y + y][self.piece_x + x] = self.piece_index + 1
        
        self.clear_lines()
        self.new_piece()
        
    def clear_lines(self):
        """Clear completed lines and update score"""
        lines_cleared = 0
        y = BOARD_HEIGHT - 1
        while y >= 0:
            if all(self.board[y]):
                lines_cleared += 1
                # Move all lines above down by one
                for row in range(y, 0, -1):
                    self.board[row] = self.board[row-1][:]
                # Clear the top line
                self.board[0] = [0] * BOARD_WIDTH
            else:
                y -= 1
                
        if lines_cleared:
            self.score += [0, 40, 100, 300, 1200][lines_cleared] * self.level
            self.level = min(10, self.score // 1000 + 1)
            
    def move(self, dx, dy):
        """Move the current piece"""
        self.piece_x += dx
        self.piece_y += dy
        
        if not self.is_valid_position():
            self.piece_x -= dx
            self.piece_y -= dy
            
            if dy > 0:  # If it was a downward move that was blocked, place the piece
                self.place_piece()
            
            return False
        return True
        
    def drop(self):
        """Drop the piece all the way down"""
        while self.move(0, 1):
            pass
        
    def draw_board(self):
        """Draw the game board and current piece"""
        self.screen.fill(BLACK)
        
        # Draw the game border
        border_rect = pygame.Rect(
            GAME_AREA_LEFT - 2, 
            GAME_AREA_TOP - 2,
            BLOCK_SIZE * BOARD_WIDTH + 4, 
            BLOCK_SIZE * BOARD_HEIGHT + 4
        )
        pygame.draw.rect(self.screen, WHITE, border_rect, 2)
        
        # Draw placed blocks on the board
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                if self.board[y][x]:
                    color_idx = self.board[y][x] - 1
                    pygame.draw.rect(
                        self.screen, 
                        COLORS[color_idx],
                        (GAME_AREA_LEFT + x * BLOCK_SIZE, 
                         GAME_AREA_TOP + y * BLOCK_SIZE,
                         BLOCK_SIZE, 
                         BLOCK_SIZE)
                    )
                    pygame.draw.rect(
                        self.screen, 
                        WHITE,
                        (GAME_AREA_LEFT + x * BLOCK_SIZE, 
                         GAME_AREA_TOP + y * BLOCK_SIZE,
                         BLOCK_SIZE, 
                         BLOCK_SIZE),
                        1
                    )
        
        # Draw the current falling piece
        for y, row in enumerate(self.current_piece):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(
                        self.screen, 
                        self.piece_color,
                        (GAME_AREA_LEFT + (self.piece_x + x) * BLOCK_SIZE, 
                         GAME_AREA_TOP + (self.piece_y + y) * BLOCK_SIZE,
                         BLOCK_SIZE, 
                         BLOCK_SIZE)
                    )
                    pygame.draw.rect(
                        self.screen, 
                        WHITE,
                        (GAME_AREA_LEFT + (self.piece_x + x) * BLOCK_SIZE, 
                         GAME_AREA_TOP + (self.piece_y + y) * BLOCK_SIZE,
                         BLOCK_SIZE, 
                         BLOCK_SIZE),
                        1
                    )
        
        # Draw score and level
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        level_text = self.font.render(f"Level: {self.level}", True, WHITE)
        self.screen.blit(score_text, (20, 20))
        self.screen.blit(level_text, (20, 60))
        
        # Draw game over message if applicable
        if self.game_over:
            game_over_text = self.font.render("GAME OVER - Press R to restart", True, RED)
            self.screen.blit(
                game_over_text, 
                (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, 
                 SCREEN_HEIGHT // 2 - game_over_text.get_height() // 2)
            )
        
        pygame.display.flip()
        
    def run(self):
        """Main game loop"""
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if not self.game_over:
                        if event.key == pygame.K_LEFT:
                            self.move(-1, 0)
                        elif event.key == pygame.K_RIGHT:
                            self.move(1, 0)
                        elif event.key == pygame.K_DOWN:
                            self.move(0, 1)
                        elif event.key == pygame.K_UP:
                            self.rotate_piece()
                        elif event.key == pygame.K_SPACE:
                            self.drop()
                    if event.key == pygame.K_r:  # Restart game
                        self.reset_game()
                    elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        running = False
            
            # Automatic falling
            current_time = time.time()
            if not self.game_over and current_time - self.last_drop_time > 0.5 / self.level:
                self.move(0, 1)
                self.last_drop_time = current_time
            
            self.draw_board()
            self.clock.tick(60)
        
        pygame.quit()


if __name__ == "__main__":
    tetris_game = Tetris()
    tetris_game.run()